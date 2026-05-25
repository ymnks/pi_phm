"""模型嵌入层与静态融合模块"""
import torch
import torch.nn as nn
import math
import sys
import os
from typing import Optional

# 添加项目根目录到Python路径以支持绝对导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import PI_PHM_Config


class DynamicFeatureEmbedding(nn.Module):
    """将动态特征嵌入到统一的特征空间"""
    
    def __init__(self, config: PI_PHM_Config, C_d: int):
        """
        Args:
            config: PI-PHM配置对象
            C_d: 动态特征通道数
        """
        super().__init__()
        self.C_d = C_d
        self.d_model = config.model.d_model
        self.T_max = 512  # 支持比lookback更长的序列
        
        # 输入投影层
        self.input_projection = nn.Linear(C_d, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(0.1)
        
        # 可学习位置编码，用正弦初始化
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.T_max, self.d_model))
        self._init_pos_encoding()
        
        # 缺失标记token
        self.mask_token = nn.Parameter(torch.zeros(self.d_model))
        
    def _init_pos_encoding(self):
        """用正弦位置编码初始化可学习参数：PE(pos,2i) = sin(pos/10000^(2i/d_model))"""
        position = torch.arange(0, self.T_max, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            (-math.log(10000.0) / self.d_model))
        pos_encoding = torch.zeros(self.T_max, self.d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pos_encoding.unsqueeze(0))
        
    def forward(self, x_dynamic: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_dynamic: (B, T, C_d) - 动态输入特征
            mask: (B, T, C_d) - 缺失掩码，True表示真实值，False表示缺失，可为None
            
        Returns:
            x_embedded: (B, T, d_model) - 嵌入后的特征
        """
        B, T, C_d = x_dynamic.shape
        assert C_d == self.C_d, f"输入通道数 {C_d} 与预期 {self.C_d} 不匹配"
        assert T <= self.T_max, f"序列长度 {T} 超过最大长度 {self.T_max}"
        
        # 处理mask为None的情况
        if mask is None:
            # 如果没有mask，假设所有值都是有效的
            mask = torch.ones_like(x_dynamic, dtype=torch.bool)
        
        # 将缺失位置设为0（已在Dataset中处理，这里作为双重保障）
        x_masked = x_dynamic * mask.float()
        
        # 投影到d_model维度
        x_projected = self.input_projection(x_masked)  # (B, T, d_model)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :T, :]  # (1, T, d_model)
        x_with_pos = x_projected + pos_enc
        
        # LayerNorm和Dropout
        x_norm = self.layer_norm(x_with_pos)
        x_dropout = self.dropout(x_norm)
        
        # 处理缺失位置：用可学习的mask_token替代
        # 首先计算每个时间步和通道是否缺失
        mask_any = mask.any(dim=-1, keepdim=True)  # (B, T, 1)
        # 扩展mask_token到批次和时间维度
        mask_token_expanded = self.mask_token.view(1, 1, -1).expand(B, T, -1)  # (B, T, d_model)
        # 对于完全缺失的时间步（所有通道都缺失），使用mask_token
        x_embedded = torch.where(mask_any, x_dropout, mask_token_expanded)
        
        return x_embedded


class StaticFeatureEncoder(nn.Module):
    """将静态地质参数编码为固定维度向量"""
    
    def __init__(self, config: PI_PHM_Config, C_geo: int):
        """
        Args:
            config: PI-PHM配置对象
            C_geo: 静态特征维度
        """
        super().__init__()
        self.C_geo = C_geo
        self.d_static = config.model.d_model
        
        # MLP编码器
        self.encoder = nn.Sequential(
            nn.Linear(C_geo, 64),
            nn.GELU(),
            nn.Linear(64, self.d_static)
        )
        
    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_static: (B, C_geo) - 静态地质参数
            
        Returns:
            static_encoded: (B, d_static) - 编码后的静态特征
        """
        B, C_geo = x_static.shape
        assert C_geo == self.C_geo, f"静态特征维度 {C_geo} 与预期 {self.C_geo} 不匹配"
        
        static_encoded = self.encoder(x_static)  # (B, d_static)
        return static_encoded


class StaticDynamicGatingFusion(nn.Module):
    """用静态编码调制动态嵌入"""
    
    def __init__(self, config: PI_PHM_Config):
        """
        Args:
            config: PI-PHM配置对象
        """
        super().__init__()
        self.d_model = config.model.d_model
        
        # 门控生成层
        self.gate_generator = nn.Linear(self.d_model, self.d_model)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_embedded: torch.Tensor, static_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_embedded: (B, T, d_model) - 动态嵌入特征
            static_encoded: (B, d_model) - 静态编码特征
            
        Returns:
            x_fused: (B, T, d_model) - 融合后的特征
        """
        B, T, d_model = x_embedded.shape
        assert d_model == self.d_model, f"嵌入维度 {d_model} 与预期 {self.d_model} 不匹配"
        assert static_encoded.shape == (B, d_model), f"静态编码形状 {static_encoded.shape} 与预期 {(B, d_model)} 不匹配"
        
        # 生成门控信号
        gate = self.sigmoid(self.gate_generator(static_encoded))  # (B, d_model)
        
        # 门控调制：x_fused = x_embedded * (1 + gate.unsqueeze(1))
        gate_expanded = gate.unsqueeze(1)  # (B, 1, d_model)
        x_fused = x_embedded * (1 + gate_expanded)  # 残差连接，保留原始嵌入
        
        return x_fused


class PhysicsAwareEmbedding(nn.Module):
    """组合动态嵌入、静态编码和融合的统一入口"""
    
    def __init__(self, config: PI_PHM_Config, C_d: int, C_geo: int):
        """
        Args:
            config: PI-PHM配置对象
            C_d: 动态特征通道数
            C_geo: 静态特征维度
        """
        super().__init__()
        self.dynamic_embedding = DynamicFeatureEmbedding(config, C_d)
        self.static_encoder = StaticFeatureEncoder(config, C_geo)
        self.fusion = StaticDynamicGatingFusion(config)
        
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dynamic: (B, T, C_d) - 动态输入特征
            x_static: (B, C_geo) - 静态地质参数
            mask: (B, T, C_d) - 缺失掩码
            
        Returns:
            x_fused: (B, T, d_model) - 融合后的特征
        """
        # 动态特征嵌入
        x_embedded = self.dynamic_embedding(x_dynamic, mask)  # (B, T, d_model)
        
        # 静态特征编码
        static_encoded = self.static_encoder(x_static)  # (B, d_model)
        
        # 静态-动态融合
        x_fused = self.fusion(x_embedded, static_encoded)  # (B, T, d_model)
        
        return x_fused


# 单元测试
if __name__ == "__main__":
    # 加载配置
    config = PI_PHM_Config()
    
    # 测试参数（这些应该从feature engineering模块获取，这里用估计值）
    B, T, C_d, C_geo = 32, 60, 89, 6
    
    # 创建随机输入
    x_dynamic = torch.randn(B, T, C_d)
    x_static = torch.randn(B, C_geo)
    mask = torch.ones(B, T, C_d, dtype=torch.bool)
    # 随机设置一些缺失值
    mask[:, :10, :] = False
    
    # 测试PhysicsAwareEmbedding
    physics_emb = PhysicsAwareEmbedding(config, C_d, C_geo)
    x_final = physics_emb(x_dynamic, x_static, mask)
    print(f"PhysicsAwareEmbedding output shape: {x_final.shape}")
    assert x_final.shape == (B, T, config.model.d_model)
    
    # 测试梯度回传
    loss = x_final.sum()
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")