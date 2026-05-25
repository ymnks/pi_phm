#!/usr/bin/env python3
"""
实验C：PatchTST + GRU + Linear head（GRU替代Mamba）
与实验B相同，但把MambaEncoder换成nn.GRU(128, 128, num_layers=2)
目标: 验证manual Mamba是否影响了增量预测
"""
import torch
import torch.nn as nn
from models.embedding import PhysicsAwareEmbedding
from models.patchtst import PatchTSTEncoder
from models.output_heads import AttentionPooling

class PatchTSTGRU(nn.Module):
    """实验C：PatchTST + GRU模型"""
    
    def __init__(self, config, dynamic_dim, static_dim=6, forecast_horizon=7, d_model=128):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        # 物理感知嵌入层（使用原始代码，传入config）
        self.embedding = PhysicsAwareEmbedding(
            config=config,
            C_d=dynamic_dim,
            C_geo=static_dim
        )
        
        # PatchTST编码器（使用原始代码）
        self.patchtst = PatchTSTEncoder(config)
        
        # GRU替代Mamba（使用PyTorch原生GRU）
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力池化（使用原始代码）
        self.pooling = AttentionPooling(d_model)
        
        # 线性输出头
        self.head = nn.Linear(d_model, forecast_horizon)
        
    def forward(self, x_dynamic, x_static, mask=None):
        """
        Args:
            x_dynamic: (batch, lookback, dynamic_dim) - 动态特征
            x_static: (batch, static_dim) - 静态特征  
            mask: (batch, lookback, dynamic_dim) - 掩码
            
        Returns:
            dict: 包含'pred_disp'键，值为(batch, 7)的预测增量
        """
        # 嵌入（需要传入mask）
        embedded = self.embedding(x_dynamic, x_static, mask)  # (batch, lookback, d_model)
        
        # PatchTST编码（返回元组，取第一个元素）
        patch_encoded_tuple = self.patchtst(embedded, mask)  # (patches, attn_weights)
        patch_encoded = patch_encoded_tuple[0]  # 只取patches
        
        # GRU处理
        gru_out, _ = self.gru(patch_encoded)  # (batch, n_patches, d_model)
        
        # 池化（返回元组，取第一个元素）
        pooled_tuple = self.pooling(gru_out)  # (h_pooled, attn_score)
        pooled = pooled_tuple[0]  # 只取h_pooled
        
        # 预测
        pred_disp = self.head(pooled)  # (batch, 7)
        
        return {'pred_disp': pred_disp}