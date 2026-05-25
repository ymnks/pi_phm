#!/usr/bin/env python3
"""
PI-PHM模型 with GRU替代Mamba
"""
import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional, Tuple
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import PI_PHM_Config
from models.embedding import PhysicsAwareEmbedding
from models.patchtst import PatchTSTEncoder
from models.physics_gate import PhysicsGateModulator
from models.output_heads import AttentionPooling, DisplacementHead, AuxiliaryDisplacementHead, RiskClassificationHead


class PIPHM_GRU(nn.Module):
    """
    Physics-Informed Patch Hybrid Model with GRU
    使用GRU替代Mamba进行全局序列建模
    """
    
    def __init__(self, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], input_channels: int = 95):
        """
        Args:
            config: PI-PHM配置对象
            feature_index_map: 物理特征在动态输入中的索引映射
            input_channels: 输入通道数，默认为95（根据特征工程结果）
        """
        super().__init__()
        self.config = config
        
        # 获取配置参数
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        n_aux = 7  # 7个位移目标（6个钻孔 + 1个GNSS）
        
        # 子模块组装
        self.embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        
        # GRU替代Mamba
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.pooling = AttentionPooling(d_model)
        self.disp_head = DisplacementHead(d_model, forecast_days)
        self.aux_disp_head = AuxiliaryDisplacementHead(d_model, forecast_days, n_aux)
        self.risk_head = RiskClassificationHead(d_model, n_classes=4)
        
    @classmethod
    def from_config(cls, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], input_channels: int = 95):
        """从配置创建模型的类方法"""
        return cls(config, feature_index_map, input_channels)
        
    def get_num_parameters(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dynamic: (B, 60, C_d) - 动态输入特征
            x_static: (B, C_geo) - 静态地质参数  
            mask: (B, 60, C_d) - 缺失掩码
            
        Returns:
            outputs: 包含所有预测和中间状态的字典
        """
        B, T, C_d = x_dynamic.shape
        assert T == self.config.model.lookback, f"Expected lookback={self.config.model.lookback}, got {T}"
        
        # 1. 嵌入与融合
        x_emb = self.embedding(x_dynamic, x_static, mask)  # (B, 60, 128)
        
        # 2. PatchTST 局部编码
        h_patches, attn_w = self.patchtst(x_emb, mask)  # (B, 11, 128)
        
        # 3. GRU 全局建模
        h_gru, gru_hidden = self.gru(h_patches)  # (B, 11, 128)
        
        # 4. 物理门控调制
        h_gated, gate_info = self.physics_gate(h_gru, x_dynamic)  # (B, 11, 128)
        
        # 5. 注意力池化
        h_pooled, pool_attn = self.pooling(h_gated)  # (B, 128)
        
        # 6. 双任务输出
        pred_disp = self.disp_head(h_pooled)       # (B, 7)
        # 添加clamp操作限制输出范围
        pred_disp = torch.clamp(pred_disp, min=-50.0, max=50.0)
        
        pred_aux = self.aux_disp_head(h_pooled)     # (B, 7, 7)
        # 对辅助任务也添加clamp
        pred_aux = torch.clamp(pred_aux, min=-50.0, max=50.0)
        
        pred_risk = self.risk_head(h_pooled)         # (B, 4)
        
        # 调试：检查DisplacementHead的输出（步骤1.3）
        if not hasattr(self, '_forward_printed'):
            print(f"pred_disp from head shape: {pred_disp.shape}")
            self._forward_printed = True
        
        # 7. 收集中间状态（用于物理约束和可解释性）
        outputs = {
            "pred_disp": pred_disp,
            "pred_aux_disp": pred_aux,
            "pred_risk_logits": pred_risk,
            "attn_weights": attn_w,
            "pool_attention": pool_attn,
            "gate_info": gate_info,
            "gru_hidden": gru_hidden,  # 替代ssm_states
        }
        return outputs


# 单元测试
if __name__ == "__main__":
    from config import PI_PHM_Config
    
    # 创建模拟的feature_index_map
    feature_index_map = {
        'velocity_indices': 19,
        'acceleration_indices': 20,
        'inverse_velocity_indices': 21,
        'piezometer_rate_indices': [22, 23, 24, 25, 26, 27],
        'seismic_rate_indices': 28,
        'rain_7d_index': 29
    }
    
    # 创建配置
    config = PI_PHM_Config()
    
    # 测试参数
    B, T, C_d, C_geo = 4, 60, 89, 6
    
    # 创建随机输入
    x_dynamic = torch.randn(B, T, C_d)
    x_static = torch.randn(B, C_geo)
    mask = torch.ones(B, T, C_d, dtype=torch.bool)
    
    # 测试完整模型
    model = PIPHM_GRU.from_config(config, feature_index_map)
    outputs = model(x_dynamic, x_static, mask)
    
    print("PIPHM_GRU Test Results:")
    print(f"Input shapes - x_dynamic: {x_dynamic.shape}, x_static: {x_static.shape}")
    print(f"Output shapes - pred_disp: {outputs['pred_disp'].shape}")
    print(f"                 pred_aux_disp: {outputs['pred_aux_disp'].shape}")  
    print(f"                 pred_risk_logits: {outputs['pred_risk_logits'].shape}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # 验证输出形状
    assert outputs['pred_disp'].shape == (B, config.model.forecast), f"主位移预测形状错误"
    assert outputs['pred_aux_disp'].shape == (B, config.model.forecast, 7), f"辅助位移预测形状错误"
    assert outputs['pred_risk_logits'].shape == (B, 4), f"风险分类预测形状错误"
    
    # 测试梯度回传
    loss = (outputs['pred_disp'].sum() + 
            outputs['pred_aux_disp'].sum() + 
            outputs['pred_risk_logits'].sum())
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")