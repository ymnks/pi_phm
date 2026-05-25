import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, Optional, Union


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, h_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_patches: (B, n_patches, d_model)
            
        Returns:
            h_pooled: (B, d_model) - 池化后的表示
            attn_score: (B, n_patches) - 注意力权重
        """
        B, n_patches, d_model = h_patches.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"
        
        # 计算注意力分数
        query_expanded = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        attn_logits = torch.sum(query_expanded * h_patches, dim=-1) / math.sqrt(d_model)  # (B, n_patches)
        attn_score = torch.softmax(attn_logits, dim=-1)  # (B, n_patches)
        
        # 加权求和
        h_pooled = torch.sum(attn_score.unsqueeze(-1) * h_patches, dim=1)  # (B, d_model)
        
        return h_pooled, attn_score


class DisplacementHead(nn.Module):
    """位移预测头"""
    
    def __init__(self, d_model: int, forecast_days: int = 7):
        super().__init__()
        self.forecast_days = forecast_days
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_days)
        )
        
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: (B, d_model)
            
        Returns:
            pred_disp: (B, forecast_days) - 未来每天的位移增量
        """
        pred_disp = self.head(h_pooled)  # (B, forecast_days)
        return pred_disp


class AuxiliaryDisplacementHead(nn.Module):
    """辅助位移预测头"""
    
    def __init__(self, d_model: int, forecast_days: int = 7, n_aux: int = 7):
        super().__init__()
        self.forecast_days = forecast_days
        self.n_aux = n_aux
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_days * n_aux)
        )
        
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: (B, d_model)
            
        Returns:
            pred_aux: (B, forecast_days, n_aux) - 6个钻孔的未来位移
        """
        B = h_pooled.shape[0]
        pred_flat = self.head(h_pooled)  # (B, forecast_days * n_aux)
        pred_aux = pred_flat.view(B, self.forecast_days, self.n_aux)  # (B, forecast_days, n_aux)
        return pred_aux


class RiskClassificationHead(nn.Module):
    """风险分类头"""
    
    def __init__(self, d_model: int, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: (B, d_model)
            
        Returns:
            pred_risk: (B, n_classes) - 风险等级logits（不加softmax）
        """
        pred_risk = self.head(h_pooled)  # (B, n_classes)
        return pred_risk


class EventDetectionHead(nn.Module):
    """蠕变爆发概率预测头"""
    
    def __init__(self, d_model: int, pos_ratio: float = 0.0836):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 初始化最后一层的偏置，使初始输出概率接近正样本基础频率
        if pos_ratio > 0 and pos_ratio < 1:
            bias_init = -math.log((1 - pos_ratio) / pos_ratio)
            self.head[-1].bias.data.fill_(bias_init)
        
    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pooled: (B, d_model)
            
        Returns:
            pred_event: (B, 1) - 蠕变爆发概率的logit
        """
        pred_event = self.head(h_pooled)  # (B, 1)
        return pred_event


class InverseVelocityPostProcessor:
    """倒速率后处理器（非nn.Module）"""
    
    @staticmethod
    def process(displacement_sequence: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Args:
            displacement_sequence: (forecast_days,) - 预测的位移序列（PyTorch tensor或numpy array）
            
        Returns:
            result: 包含T_fail_days, confidence, R2的字典
        """
        import numpy as np
        from scipy.stats import linregress
        
        # 处理输入类型：支持PyTorch tensor和numpy array
        if isinstance(displacement_sequence, torch.Tensor):
            # PyTorch tensor: 转换为numpy数组
            disp_np = displacement_sequence.detach().cpu().numpy()
        elif isinstance(displacement_sequence, np.ndarray):
            # 已经是numpy数组
            disp_np = displacement_sequence
        else:
            # 其他类型：尝试转换为numpy数组
            disp_np = np.array(displacement_sequence)
            
        time_steps = np.arange(len(disp_np))
        
        # 计算速度（一阶差分）
        velocity = np.diff(disp_np, prepend=disp_np[0])
        velocity = np.maximum(velocity, 1e-8)  # 避免除零
        
        # 计算倒速率
        inverse_velocity = 1.0 / velocity
        
        # 线性回归拟合倒速率
        slope, intercept, r_value, p_value, std_err = linregress(time_steps, inverse_velocity)
        r_squared = r_value ** 2
        
        result = {
            "R2": float(r_squared),
            "confidence": 0.0,
            "T_fail_days": None
        }
        
        # 只有当斜率为负时才有物理意义（加速蠕变）
        if slope < 0:
            T_fail = -intercept / slope
            # 检查外推时间是否在合理范围（1-365天）
            if 1 <= T_fail <= 365:
                result["T_fail_days"] = float(T_fail)
                # 置信度基于R²和斜率显著性
                confidence = r_squared * (1 - p_value)
                result["confidence"] = min(confidence, 1.0)
        
        return result


# 单元测试
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 测试参数
    B, n_patches, d_model, forecast_days, n_aux = 4, 11, 128, 7, 6
    
    # 测试AttentionPooling
    h_patches = torch.randn(B, n_patches, d_model)
    pooling = AttentionPooling(d_model)
    h_pooled, attn_score = pooling(h_patches)
    print(f"AttentionPooling - Input: {h_patches.shape}, Output: {h_pooled.shape}, Attn: {attn_score.shape}")
    assert h_pooled.shape == (B, d_model)
    assert attn_score.shape == (B, n_patches)
    
    # 测试DisplacementHead
    disp_head = DisplacementHead(d_model, forecast_days)
    pred_disp = disp_head(h_pooled)
    print(f"DisplacementHead - Input: {h_pooled.shape}, Output: {pred_disp.shape}")
    assert pred_disp.shape == (B, forecast_days)
    
    # 测试AuxiliaryDisplacementHead
    aux_head = AuxiliaryDisplacementHead(d_model, forecast_days, n_aux)
    pred_aux = aux_head(h_pooled)
    print(f"AuxiliaryDisplacementHead - Input: {h_pooled.shape}, Output: {pred_aux.shape}")
    assert pred_aux.shape == (B, forecast_days, n_aux)
    
    # 测试RiskClassificationHead
    risk_head = RiskClassificationHead(d_model, n_classes=4)
    pred_risk = risk_head(h_pooled)
    print(f"RiskClassificationHead - Input: {h_pooled.shape}, Output: {pred_risk.shape}")
    assert pred_risk.shape == (B, 4)
    
    # 测试InverseVelocityPostProcessor
    sample_disp = torch.tensor([0.1, 0.2, 0.4, 0.7, 1.1, 1.6, 2.2])  # 加速序列
    result = InverseVelocityPostProcessor.process(sample_disp)
    print(f"InverseVelocityPostProcessor - Result: {result}")
    assert "T_fail_days" in result
    assert "confidence" in result
    assert "R2" in result
    
    # 测试梯度回传
    loss = pred_disp.sum() + pred_aux.sum() + pred_risk.sum()
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")