#!/usr/bin/env python3
"""
实验A：仅用最简单的线性模型
输入: x_dynamic最后一天的GNSS速度
模型: Linear(1 → 7)
目标: 预测7天增量
期望MAE: 接近Persistence基线
"""
import torch
import torch.nn as nn

class LinearBaseline(nn.Module):
    """实验A：线性基线模型"""
    
    def __init__(self, forecast_horizon=7):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        # 线性层：1个输入（GNSS速度）→ 7个输出（7天增量预测）
        self.linear = nn.Linear(1, forecast_horizon)
        
    def forward(self, x_dynamic, x_static=None, mask=None):
        """
        Args:
            x_dynamic: (batch, lookback, features) - 动态特征
            x_static: (batch, static_features) - 静态特征（此处不使用）
            mask: (batch, lookback) - 掩码（此处不使用）
            
        Returns:
            dict: 包含'pred_disp'键，值为(batch, 7)的预测增量
        """
        # 取最后一天的GNSS速度（假设GNSS_12H是第0个特征）
        # x_dynamic shape: (batch, lookback, features)
        last_gnss_velocity = x_dynamic[:, -1, 0:1]  # (batch, 1)
        
        # 线性预测7天增量
        pred_disp = self.linear(last_gnss_velocity)  # (batch, 7)
        
        return {'pred_disp': pred_disp}