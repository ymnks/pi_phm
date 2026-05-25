#!/usr/bin/env python3
"""
Task Decoupling Models for PI-PHM
实现debug9.md要求的E1-E5实验模型变体
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Dict, Union, List, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.embedding import PhysicsAwareEmbedding
from models.patchtst import PatchTSTEncoder  
from models.mamba_block import MambaEncoder
from models.physics_gate import PhysicsGateModulator
from models.output_heads import AttentionPooling, DisplacementHead, RiskClassificationHead, EventDetectionHead


class DisplacementOnlyModel(nn.Module):
    """
    E1: Displacement-only model
    只训练位移增量预测，不训练event head和risk head
    """
    
    def __init__(self, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], input_channels: int = 103):
        super().__init__()
        self.config = config
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        
        # 共享组件（与原PI-PHM保持一致）
        self.embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        self.mamba = MambaEncoder(config)
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.pooling = AttentionPooling(d_model)
        self.disp_head = DisplacementHead(d_model, forecast_days)
        
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, C_d = x_dynamic.shape
        assert T == self.config.model.lookback, f"Expected lookback={self.config.model.lookback}, got {T}"
        
        # 前向传播（与原PI-PHM保持一致）
        x_emb = self.embedding(x_dynamic, x_static, mask)
        h_patches, attn_w = self.patchtst(x_emb, mask)
        h_mamba, ssm_states = self.mamba(h_patches)
        h_gated, gate_info = self.physics_gate(h_mamba, x_dynamic)
        h_pooled, pool_attn = self.pooling(h_gated)
        pred_disp = self.disp_head(h_pooled)
        pred_disp = torch.clamp(pred_disp, min=-50.0, max=50.0)
        
        return {
            'pred_disp': pred_disp,
            'attn_weights': attn_w,
            'gate_info': gate_info
        }


class EventOnlyModel(nn.Module):
    """
    E2: Event-only model
    只训练creep burst事件检测，不训练displacement head和risk head
    
    支持两种输入模式：
    - full_input: 使用全部动态特征（包括GNSS）
    - borehole_focused: 只使用钻孔位移、孔压、微震、气象特征，排除GNSS
    """
    
    def __init__(self, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], 
                 input_channels: int = 103, use_gnss: bool = True, gnss_indices: Optional[List[int]] = None):
        super().__init__()
        self.config = config
        self.use_gnss = use_gnss
        self.gnss_indices = gnss_indices if gnss_indices is not None else []
        
        d_model = config.model.d_model
        
        # 计算实际输入通道数
        if not use_gnss and gnss_indices:
            actual_input_channels = input_channels - len(gnss_indices)
            # 创建新的feature_index_map，排除GNSS特征
            self.filtered_feature_index_map = self._filter_feature_index_map(feature_index_map, gnss_indices)
        else:
            actual_input_channels = input_channels
            self.filtered_feature_index_map = feature_index_map
            
        # 共享组件（与原PI-PHM保持一致）
        self.embedding = PhysicsAwareEmbedding(config, C_d=actual_input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        self.mamba = MambaEncoder(config)
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=self.filtered_feature_index_map)
        self.pooling = AttentionPooling(d_model)
        self.event_head = EventDetectionHead(d_model)
        
    def _filter_feature_index_map(self, original_map: Dict[str, Union[int, List[int]]], gnss_indices: List[int]) -> Dict[str, Union[int, List[int]]]:
        """过滤feature_index_map，移除GNSS特征并重新映射索引"""
        filtered_map = {}
        gnss_set = set(gnss_indices)
        
        for key, value in original_map.items():
            if isinstance(value, int):
                if value not in gnss_set:
                    # 计算新索引（减去前面的GNSS特征数量）
                    new_index = value - sum(1 for idx in gnss_indices if idx < value)
                    filtered_map[key] = new_index
            elif isinstance(value, list):
                # 过滤列表中的GNSS索引
                filtered_indices = [idx for idx in value if idx not in gnss_set]
                if filtered_indices:
                    # 重新映射索引
                    new_indices = []
                    for idx in filtered_indices:
                        new_idx = idx - sum(1 for g_idx in gnss_indices if g_idx < idx)
                        new_indices.append(new_idx)
                    filtered_map[key] = new_indices
        
        return filtered_map
    
    def _filter_gnss_features(self, x_dynamic: torch.Tensor) -> torch.Tensor:
        """过滤掉GNSS特征"""
        if self.use_gnss or not self.gnss_indices:
            return x_dynamic
            
        # 检查索引是否有效
        valid_gnss_indices = [idx for idx in self.gnss_indices if idx < x_dynamic.shape[-1]]
        if len(valid_gnss_indices) != len(self.gnss_indices):
            self.gnss_indices = valid_gnss_indices
            
        # 创建非GNSS特征的掩码
        all_indices = set(range(x_dynamic.shape[-1]))
        non_gnss_indices = sorted(list(all_indices - set(self.gnss_indices)))
        
        # 选择非GNSS特征
        x_filtered = x_dynamic[:, :, non_gnss_indices]
        return x_filtered
    
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, C_d = x_dynamic.shape
        assert T == self.config.model.lookback, f"Expected lookback={self.config.model.lookback}, got {T}"
        
        # 过滤GNSS特征（如果需要）
        x_processed = self._filter_gnss_features(x_dynamic)
        
        # 更新mask（如果提供了mask）
        if mask is not None:
            mask_processed = self._filter_gnss_features(mask)
        else:
            mask_processed = None
            
        # 前向传播
        x_emb = self.embedding(x_processed, x_static, mask_processed)
        h_patches, attn_w = self.patchtst(x_emb, mask_processed)
        h_mamba, ssm_states = self.mamba(h_patches)
        h_gated, gate_info = self.physics_gate(h_mamba, x_processed)
        h_pooled, pool_attn = self.pooling(h_gated)
        pred_event = self.event_head(h_pooled)
        
        return {
            'pred_event_logits': pred_event,
            'attn_weights': attn_w,
            'gate_info': gate_info
        }


class FullSharedMultitaskModel(nn.Module):
    """
    E3: Full-shared multitask model
    这是当前PI-PHM风格的基线，共享embedding + shared backbone + 多头输出
    """
    
    def __init__(self, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], input_channels: int = 103):
        super().__init__()
        self.config = config
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        n_aux = 7  # 7个位移目标
        
        # 完全共享的组件
        self.embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        self.mamba = MambaEncoder(config)
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.pooling = AttentionPooling(d_model)
        
        # 多任务头
        self.disp_head = DisplacementHead(d_model, forecast_days)
        from models.output_heads import AuxiliaryDisplacementHead
        self.aux_disp_head = AuxiliaryDisplacementHead(d_model, forecast_days, n_aux)
        self.risk_head = RiskClassificationHead(d_model, n_classes=4)
        self.event_head = EventDetectionHead(d_model, pos_ratio=0.0836)
        
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, C_d = x_dynamic.shape
        assert T == self.config.model.lookback, f"Expected lookback={self.config.model.lookback}, got {T}"
        
        # 共享前向传播
        x_emb = self.embedding(x_dynamic, x_static, mask)
        h_patches, attn_w = self.patchtst(x_emb, mask)
        h_mamba, ssm_states = self.mamba(h_patches)
        h_gated, gate_info = self.physics_gate(h_mamba, x_dynamic)
        h_pooled, pool_attn = self.pooling(h_gated)
        
        # 多任务输出
        pred_disp = self.disp_head(h_pooled)
        pred_disp = torch.clamp(pred_disp, min=-50.0, max=50.0)
        
        pred_aux = self.aux_disp_head(h_pooled)
        pred_aux = torch.clamp(pred_aux, min=-50.0, max=50.0)
        
        pred_risk = self.risk_head(h_pooled)
        pred_event = self.event_head(h_pooled)
        
        return {
            'pred_disp': pred_disp,
            'pred_aux_disp': pred_aux,
            'pred_risk_logits': pred_risk,
            'pred_event_logits': pred_event,
            'attn_weights': attn_w,
            'gate_info': gate_info
        }


class PartialSharedMultitaskModel(nn.Module):
    """
    E4: Partial-shared multitask model
    测试"只共享浅层表示，但分开高层backbone"是否更合理
    
    结构：
    - 共享最浅层embedding
    - 从embedding之后分成两个分支：
      - 分支A（位移分支）：PatchTST + GRU + displacement head
      - 分支B（事件分支）：PatchTST + GRU + event head (+ risk head)
    """
    
    def __init__(self, config: 'PI_PHM_Config', feature_index_map: Dict[str, Union[int, List[int]]], input_channels: int = 103):
        super().__init__()
        self.config = config
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        
        # 共享的embedding层
        self.shared_embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        
        # 位移分支（独立的backbone）
        self.disp_patchtst = PatchTSTEncoder(config)
        self.disp_mamba = MambaEncoder(config)
        self.disp_physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.disp_pooling = AttentionPooling(d_model)
        self.disp_head = DisplacementHead(d_model, forecast_days)
        
        # 事件分支（独立的backbone）
        self.event_patchtst = PatchTSTEncoder(config)
        self.event_mamba = MambaEncoder(config)
        self.event_physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.event_pooling = AttentionPooling(d_model)
        self.event_head = EventDetectionHead(d_model, pos_ratio=0.0836)
        self.risk_head = RiskClassificationHead(d_model, n_classes=4)
        
    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, C_d = x_dynamic.shape
        assert T == self.config.model.lookback, f"Expected lookback={self.config.model.lookback}, got {T}"
        
        # 共享embedding
        x_emb = self.shared_embedding(x_dynamic, x_static, mask)
        
        # 位移分支
        disp_h_patches, disp_attn_w = self.disp_patchtst(x_emb, mask)
        disp_h_mamba, disp_ssm_states = self.disp_mamba(disp_h_patches)
        disp_h_gated, disp_gate_info = self.disp_physics_gate(disp_h_mamba, x_dynamic)
        disp_h_pooled, disp_pool_attn = self.disp_pooling(disp_h_gated)
        pred_disp = self.disp_head(disp_h_pooled)
        pred_disp = torch.clamp(pred_disp, min=-50.0, max=50.0)
        
        # 事件分支
        event_h_patches, event_attn_w = self.event_patchtst(x_emb, mask)
        event_h_mamba, event_ssm_states = self.event_mamba(event_h_patches)
        event_h_gated, event_gate_info = self.event_physics_gate(event_h_mamba, x_dynamic)
        event_h_pooled, event_pool_attn = self.event_pooling(event_h_gated)
        pred_event = self.event_head(event_h_pooled)
        pred_risk = self.risk_head(event_h_pooled)
        
        return {
            'pred_disp': pred_disp,
            'pred_event_logits': pred_event,
            'pred_risk_logits': pred_risk,
            'disp_attn_weights': disp_attn_w,
            'event_attn_weights': event_attn_w,
            'disp_gate_info': disp_gate_info,
            'event_gate_info': event_gate_info
        }


def create_task_decoupling_model(model_variant: str, config: 'PI_PHM_Config', 
                               feature_index_map: Dict[str, Union[int, List[int]]], 
                               input_channels: int = 103,
                               **kwargs) -> nn.Module:
    """
    工厂函数：根据model_variant创建对应的模型
    
    Args:
        model_variant: 模型变体类型
            - "disp_only": E1
            - "event_only_full": E2a  
            - "event_only_borehole": E2b
            - "full_shared": E3
            - "partial_shared": E4
            - "dual_model": E5 (需要特殊处理)
        config: 配置对象
        feature_index_map: 特征索引映射
        input_channels: 输入通道数
        **kwargs: 额外参数（如gnss_indices等）
        
    Returns:
        对应的模型实例
    """
    if model_variant == "disp_only":
        return DisplacementOnlyModel(config, feature_index_map, input_channels)
    elif model_variant == "event_only_full":
        return EventOnlyModel(config, feature_index_map, input_channels, use_gnss=True)
    elif model_variant == "event_only_borehole":
        gnss_indices = kwargs.get('gnss_indices', [])
        return EventOnlyModel(config, feature_index_map, input_channels, 
                            use_gnss=False, gnss_indices=gnss_indices)
    elif model_variant == "full_shared":
        return FullSharedMultitaskModel(config, feature_index_map, input_channels)
    elif model_variant == "partial_shared":
        return PartialSharedMultitaskModel(config, feature_index_map, input_channels)
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")


def get_gnss_feature_indices(feature_names: List[str]) -> List[int]:
    """
    从特征名称列表中识别GNSS相关特征的索引
    
    Args:
        feature_names: 特征名称列表
        
    Returns:
        GNSS特征的索引列表
    """
    gnss_indices = []
    gnss_keywords = ['GNSS', 'gps', 'GPS']
    
    for i, name in enumerate(feature_names):
        if any(keyword in name for keyword in gnss_keywords):
            gnss_indices.append(i)
    
    return gnss_indices