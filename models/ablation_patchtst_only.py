#!/usr/bin/env python3
"""
实验B：PatchTST + Linear head（无Mamba，无Physics Gate，无静态流）
输入: x_dynamic (60, C_d)
模型: PhysicsAwareEmbedding → PatchTST → AttentionPooling → Linear(128→7)
目标: 仅位移增量回归（无风险分类，无物理损失）
期望MAE: 应比实验A好
"""
import torch
import torch.nn as nn
from models.embedding import PhysicsAwareEmbedding
from models.patchtst import PatchTSTEncoder
from models.output_heads import AttentionPooling

class PatchTSTOnly(nn.Module):
    """实验B：PatchTST-only模型"""
    
    def __init__(self, config, dynamic_dim, static_dim=6, forecast_horizon=7):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.forecast_horizon = forecast_horizon
        
        # 物理感知嵌入层（使用原始代码，传入config）
        self.embedding = PhysicsAwareEmbedding(
            config=config,
            C_d=dynamic_dim,
            C_geo=static_dim
        )
        
        # PatchTST编码器（使用原始代码）
        self.patchtst = PatchTSTEncoder(config)
        
        # 注意力池化（使用原始代码）
        self.pooling = AttentionPooling(config.model.d_model)
        
        # 线性输出头
        self.head = nn.Linear(config.model.d_model, forecast_horizon)
        
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
        encoded_tuple = self.patchtst(embedded, mask)  # (patches, attn_weights)
        encoded = encoded_tuple[0]  # 只取patches
        
        # 池化（返回元组，取第一个元素）
        pooled_tuple = self.pooling(encoded)  # (h_pooled, attn_score)
        pooled = pooled_tuple[0]  # 只取h_pooled
        
        # 预测
        pred_disp = self.head(pooled)  # (batch, 7)
        
        return {'pred_disp': pred_disp}