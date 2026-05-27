import torch
import torch.nn as nn
from typing import Dict, Union, List

from config import PI_PHM_Config
from models.embedding import PhysicsAwareEmbedding
from models.patchtst import PatchTSTEncoder
from models.physics_gate import PhysicsGateModulator
from models.output_heads import (
    AttentionPooling,
    DisplacementHead,
    AuxiliaryDisplacementHead,
    RiskClassificationHead,
    EventDetectionHead,
)


class PIPHM_GRU_Full(nn.Module):
    """AAAI GROUP1 fair-comparison GRU backbone.

    Keeps the full PI-PHM pipeline unchanged except replacing the global backbone
    with a 2-layer GRU. Event head is retained for protocol-aligned comparison.
    """

    def __init__(
        self,
        config: PI_PHM_Config,
        feature_index_map: Dict[str, Union[int, List[int]]],
        input_channels: int = 103,
    ):
        super().__init__()
        self.config = config
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        n_aux = 7

        self.embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.pooling = AttentionPooling(d_model)
        self.disp_head = DisplacementHead(d_model, forecast_days)
        self.aux_disp_head = AuxiliaryDisplacementHead(d_model, forecast_days, n_aux)
        self.risk_head = RiskClassificationHead(d_model, n_classes=4)
        self.event_head = EventDetectionHead(d_model)

    @classmethod
    def from_config(cls, config, feature_index_map, input_channels=103):
        return cls(config, feature_index_map, input_channels)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None):
        x_emb = self.embedding(x_dynamic, x_static, mask)
        h_patches, attn_w = self.patchtst(x_emb, mask)
        h_backbone, gru_hidden = self.gru(h_patches)
        h_gated, gate_info = self.physics_gate(h_backbone, x_dynamic)
        h_pooled, pool_attn = self.pooling(h_gated)

        pred_disp = torch.clamp(self.disp_head(h_pooled), min=-50.0, max=50.0)
        pred_aux = torch.clamp(self.aux_disp_head(h_pooled), min=-50.0, max=50.0)
        pred_risk = self.risk_head(h_pooled)
        pred_event = self.event_head(h_pooled)

        return {
            'pred_disp': pred_disp,
            'pred_aux_disp': pred_aux,
            'pred_risk_logits': pred_risk,
            'pred_event_logits': pred_event,
            'attn_weights': attn_w,
            'pool_attention': pool_attn,
            'gate_info': gate_info,
            'gru_hidden': gru_hidden,
        }


class PIPHM_PatchTSTOnly_Full(nn.Module):
    """AAAI GROUP1 fair-comparison PatchTST-only backbone.

    Removes the global backbone only. All other PI-PHM components are retained so
    the comparison isolates the contribution of global sequence modeling.
    """

    def __init__(
        self,
        config: PI_PHM_Config,
        feature_index_map: Dict[str, Union[int, List[int]]],
        input_channels: int = 103,
    ):
        super().__init__()
        self.config = config
        d_model = config.model.d_model
        forecast_days = config.model.forecast
        n_aux = 7

        self.embedding = PhysicsAwareEmbedding(config, C_d=input_channels, C_geo=6)
        self.patchtst = PatchTSTEncoder(config)
        self.physics_gate = PhysicsGateModulator(d_model, n_patches=11, feature_index_map=feature_index_map)
        self.pooling = AttentionPooling(d_model)
        self.disp_head = DisplacementHead(d_model, forecast_days)
        self.aux_disp_head = AuxiliaryDisplacementHead(d_model, forecast_days, n_aux)
        self.risk_head = RiskClassificationHead(d_model, n_classes=4)
        self.event_head = EventDetectionHead(d_model)

    @classmethod
    def from_config(cls, config, feature_index_map, input_channels=103):
        return cls(config, feature_index_map, input_channels)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor = None):
        x_emb = self.embedding(x_dynamic, x_static, mask)
        h_patches, attn_w = self.patchtst(x_emb, mask)
        h_gated, gate_info = self.physics_gate(h_patches, x_dynamic)
        h_pooled, pool_attn = self.pooling(h_gated)

        pred_disp = torch.clamp(self.disp_head(h_pooled), min=-50.0, max=50.0)
        pred_aux = torch.clamp(self.aux_disp_head(h_pooled), min=-50.0, max=50.0)
        pred_risk = self.risk_head(h_pooled)
        pred_event = self.event_head(h_pooled)

        return {
            'pred_disp': pred_disp,
            'pred_aux_disp': pred_aux,
            'pred_risk_logits': pred_risk,
            'pred_event_logits': pred_event,
            'attn_weights': attn_w,
            'pool_attention': pool_attn,
            'gate_info': gate_info,
        }
