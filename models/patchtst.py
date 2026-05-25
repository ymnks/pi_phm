import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PatchSplitter(nn.Module):
    """将时间序列切分为patch并进行投影"""
    
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch投影层：将每个patch展平后投影到d_model维度
        self.patch_projection = nn.Linear(patch_len * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的patch位置编码
        # 最大支持的patch数量（对于T=60, patch_len=10, stride=5 → n_patches=11）
        max_patches = 512
        self.patch_pos_encoding = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self._init_patch_pos_encoding(max_patches)
        
    def _init_patch_pos_encoding(self, max_patches: int):
        """用正弦位置编码初始化patch位置编码"""
        position = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            (-math.log(10000.0) / self.d_model))
        pos_encoding = torch.zeros(max_patches, self.d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.patch_pos_encoding.data.copy_(pos_encoding.unsqueeze(0))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model) - 输入嵌入
            mask: (B, T, C_d) 或 None - 原始mask
            
        Returns:
            patches_projected: (B, n_patches, d_model) - 投影后的patch
            patch_mask: (B, n_patches) - patch级别的mask
            patch_indices: (n_patches,) - patch在原始序列中的起始位置
        """
        B, T, d_model = x.shape
        assert d_model == self.d_model, f"输入维度 {d_model} 与预期 {self.d_model} 不匹配"
        
        # 计算patch数量
        if T < self.patch_len:
            # 如果序列太短，padding到patch_len
            padding = self.patch_len - T
            x = torch.cat([x, torch.zeros(B, padding, d_model, device=x.device)], dim=1)
            T = self.patch_len
            if mask is not None:
                mask_padding = torch.zeros(B, padding, mask.shape[-1], device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, mask_padding], dim=1)
        
        # 使用unfold操作切分patch
        # x_unfold: (B, d_model, n_patches, patch_len)
        x_unfold = x.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        n_patches = x_unfold.size(-2)
        
        # 转换为 (B, n_patches, patch_len, d_model)
        patches = x_unfold.permute(0, 2, 3, 1)
        
        # 展平每个patch并投影
        patches_flat = patches.reshape(B, n_patches, -1)  # (B, n_patches, patch_len * d_model)
        patches_projected = self.patch_projection(patches_flat)  # (B, n_patches, d_model)
        
        # 添加patch位置编码
        pos_enc = self.patch_pos_encoding[:, :n_patches, :]  # (1, n_patches, d_model)
        patches_projected = patches_projected + pos_enc
        
        # Dropout
        patches_projected = self.dropout(patches_projected)
        
        # 计算patch级别的mask
        patch_mask = torch.ones(B, n_patches, device=x.device, dtype=torch.bool)
        if mask is not None:
            # mask: (B, T, C_d) -> 需要聚合到patch级别
            # 首先计算每个时间步是否有效（至少有一个通道有效）
            time_step_valid = mask.any(dim=-1)  # (B, T)
            
            # 对每个patch，检查超过50%的时间步是否有效
            for i in range(n_patches):
                start_idx = i * self.stride
                end_idx = start_idx + self.patch_len
                if end_idx > T:
                    # 处理最后一个patch可能超出的情况
                    end_idx = T
                    valid_steps = time_step_valid[:, start_idx:end_idx]
                    valid_ratio = valid_steps.float().mean(dim=-1)
                else:
                    valid_steps = time_step_valid[:, start_idx:end_idx]
                    valid_ratio = valid_steps.float().mean(dim=-1)
                
                # 如果有效比例 <= 0.5，则标记为masked
                patch_mask[:, i] = valid_ratio > 0.5
        
        # 计算patch在原始序列中的起始位置（用于可视化）
        patch_indices = torch.arange(0, n_patches * self.stride, self.stride)[:n_patches]
        
        return patches_projected, patch_mask, patch_indices


class PreNormMultiHeadAttention(nn.Module):
    """Pre-Norm风格的多头自注意力"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} 必须被 n_heads {n_heads} 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_patches, d_model) - 输入
            mask: (B, n_patches) 或 None - patch级别的mask
            
        Returns:
            output: (B, n_patches, d_model) - 输出
            attn_weights: (B, n_heads, n_patches, n_patches) - 注意力权重
        """
        B, n_patches, d_model = x.shape
        assert d_model == self.d_model, f"输入维度 {d_model} 与预期 {self.d_model} 不匹配"
        
        # Pre-Norm
        x_norm = self.layer_norm(x)
        
        # 线性投影
        q = self.q_proj(x_norm)  # (B, n_patches, d_model)
        k = self.k_proj(x_norm)  # (B, n_patches, d_model)
        v = self.v_proj(x_norm)  # (B, n_patches, d_model)
        
        # 分头
        q = q.view(B, n_patches, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, n_patches, head_dim)
        k = k.view(B, n_patches, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, n_patches, head_dim)
        v = v.view(B, n_patches, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, n_patches, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, n_patches, n_patches)
        
        # 应用mask
        if mask is not None:
            # mask: (B, n_patches) -> 扩展到 (B, 1, 1, n_patches)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, n_patches)
            # 将masked位置的注意力分数设为-inf
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, n_heads, n_patches, n_patches)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)  # (B, n_heads, n_patches, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, n_patches, d_model)  # (B, n_patches, d_model)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attn_weights


class PreNormFeedForward(nn.Module):
    """Pre-Norm风格的前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, d_model) - 输入
            
        Returns:
            output: (B, n_patches, d_model) - 输出
        """
        x_norm = self.layer_norm(x)
        output = self.ffn(x_norm)
        return output


class PatchTSTEncoderLayer(nn.Module):
    """单个PatchTST编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = PreNormMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PreNormFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_patches, d_model) - 输入
            mask: (B, n_patches) 或 None - patch级别的mask
            
        Returns:
            output: (B, n_patches, d_model) - 输出
            attn_weights: (B, n_heads, n_patches, n_patches) - 注意力权重
        """
        # Self-Attention + Residual Connection
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + attn_output
        
        # Feed-Forward + Residual Connection
        ffn_output = self.ffn(x)
        output = x + ffn_output
        
        return output, attn_weights


class PatchTSTEncoder(nn.Module):
    """PatchTST编码器主类"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        """
        Args:
            config: PI-PHM配置对象
        """
        super().__init__()
        self.d_model = config.model.d_model
        self.patch_len = config.model.patch_len
        self.stride = config.model.stride
        self.n_layers = config.model.n_patchtst_layers
        self.n_heads = config.model.n_heads
        self.d_ff = 2 * self.d_model  # 前馈网络隐藏维度
        self.dropout = 0.1
        
        # 验证head_dim整除
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} 必须被 n_heads {self.n_heads} 整除"
        self.head_dim = self.d_model // self.n_heads
        
        # Patch分割层
        self.patch_splitter = PatchSplitter(self.d_model, self.patch_len, self.stride, self.dropout)
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
    def forward(self, x_embedded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_embedded: (B, T, d_model) - 嵌入后的输入，T=60
            mask: (B, T, C_d) 或 None - 原始mask
            
        Returns:
            h_patches: (B, n_patches, d_model) - 编码后的patch表示
            attn_weights: (B, n_heads, n_patches, n_patches) - 最后一层的注意力权重
        """
        B, T, d_model = x_embedded.shape
        assert d_model == self.d_model, f"输入维度 {d_model} 与预期 {self.d_model} 不匹配"
        
        # Patch分割
        patches, patch_mask, patch_indices = self.patch_splitter(x_embedded, mask)
        h_patches = patches
        
        # 通过所有编码器层
        final_attn_weights = None
        for layer in self.layers:
            h_patches, attn_weights = layer(h_patches, patch_mask)
            final_attn_weights = attn_weights  # 保存最后一层的注意力权重
            
        return h_patches, final_attn_weights


# 单元测试
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from config import PI_PHM_Config
    
    # 创建配置
    config = PI_PHM_Config()
    
    # 测试参数
    B, T, d_model = 2, 60, 128
    
    # 创建随机输入
    x_embedded = torch.randn(B, T, d_model)
    mask = torch.ones(B, T, 89, dtype=torch.bool)  # C_d=89
    
    # 测试PatchTSTEncoder
    encoder = PatchTSTEncoder(config)
    h_patches, attn_weights = encoder(x_embedded, mask)
    
    print(f"Input shape: {x_embedded.shape}")
    print(f"Output shape: {h_patches.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 验证输出形状
    expected_n_patches = (T - config.model.patch_len) // config.model.stride + 1
    assert h_patches.shape == (B, expected_n_patches, d_model), f"输出形状错误: {h_patches.shape} != {(B, expected_n_patches, d_model)}"
    assert attn_weights.shape == (B, config.model.n_heads, expected_n_patches, expected_n_patches), f"注意力权重形状错误"
    
    # 测试梯度回传
    loss = h_patches.sum() + attn_weights.sum()
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")