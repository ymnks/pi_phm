import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class PhysicsGateModulator(nn.Module):
    """物理感知门控调制模块"""
    
    def __init__(self, d_model: int, n_patches: int, feature_index_map: Dict[str, Union[int, List[int]]]):
        """
        Args:
            d_model: 模型维度
            n_patches: patch数量
            feature_index_map: 物理特征在动态输入中的索引映射
                - velocity_indices: GNSS_12H_velocity 列索引 (int)
                - acceleration_indices: GNSS_12H_acceleration 列索引 (int)  
                - inverse_velocity_indices: GNSS_12H_inverse_velocity 列索引 (int)
                - piezometer_rate_indices: 6个孔压变化率列索引列表 (List[int])
                - seismic_rate_indices: seismic_total_rate 列索引 (int)
                - rain_7d_index: rain_7d 列索引 (int)
        """
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        
        # 验证feature_index_map的完整性
        required_keys = [
            'velocity_indices', 'acceleration_indices', 'inverse_velocity_indices',
            'piezometer_rate_indices', 'seismic_rate_indices', 'rain_7d_index'
        ]
        for key in required_keys:
            assert key in feature_index_map, f"Missing required key '{key}' in feature_index_map"
            
        self.feature_index_map = feature_index_map
        
        # 维度级门控生成器 (8 -> d_model)
        self.dim_gate_generator = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.Sigmoid()
        )
        
        # Patch级门控生成器 (8 -> n_patches)
        self.patch_gate_generator = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, n_patches),
            nn.Sigmoid()
        )
        
    def extract_physics_summary(self, x_dynamic_raw: torch.Tensor) -> torch.Tensor:
        """
        从原始动态输入中提取物理状态摘要
        
        Args:
            x_dynamic_raw: (B, T=60, C_d) - 原始归一化后的动态输入
            
        Returns:
            physics_summary: (B, 8) - 物理状态摘要向量
                [mean_velocity, mean_acceleration, mean_inverse_velocity,
                 mean_rain_7d, mean_piezometer_rate, max_piezometer_rate,
                 mean_seismic_rate, max_seismic_rate]
        """
        B, T, C_d = x_dynamic_raw.shape
        assert T == 60, f"Expected T=60, got {T}"
        
        # 提取最后7天的数据
        last_7_days = x_dynamic_raw[:, -7:, :]  # (B, 7, C_d)
        
        # 提取各物理量
        vel_idx = self.feature_index_map['velocity_indices']
        acc_idx = self.feature_index_map['acceleration_indices']
        inv_vel_idx = self.feature_index_map['inverse_velocity_indices']
        piezo_indices = self.feature_index_map['piezometer_rate_indices']
        seismic_idx = self.feature_index_map['seismic_rate_indices']
        rain_idx = self.feature_index_map['rain_7d_index']
        
        # creep_state (3维) - 处理多索引情况
        if isinstance(vel_idx, list):
            vel_values = last_7_days[:, :, vel_idx]  # (B, 7, num_vel)
            mean_velocity = vel_values.mean(dim=[1, 2])  # (B,) - 所有速度传感器的均值
        else:
            mean_velocity = last_7_days[:, :, vel_idx].mean(dim=1)  # (B,)
            
        if isinstance(acc_idx, list):
            acc_values = last_7_days[:, :, acc_idx]  # (B, 7, num_acc)
            mean_acceleration = acc_values.mean(dim=[1, 2])  # (B,)
        else:
            mean_acceleration = last_7_days[:, :, acc_idx].mean(dim=1)  # (B,)
            
        if isinstance(inv_vel_idx, list):
            inv_vel_values = last_7_days[:, :, inv_vel_idx]  # (B, 7, num_inv_vel)
            mean_inverse_velocity = inv_vel_values.mean(dim=[1, 2])  # (B,)
        else:
            mean_inverse_velocity = last_7_days[:, :, inv_vel_idx].mean(dim=1)  # (B,)
        
        # hydro_state (3维)
        mean_rain_7d = last_7_days[:, :, rain_idx].mean(dim=1)  # (B,)
        
        # 处理孔压变化率（可能是多个索引）
        if isinstance(piezo_indices, list):
            piezo_rates = last_7_days[:, :, piezo_indices]  # (B, 7, num_piezo)
            mean_piezometer_rate = piezo_rates.mean(dim=[1, 2])  # (B,) - 所有孔压传感器的均值
            max_piezometer_rate = piezo_rates.max(dim=2)[0].max(dim=1)[0]  # (B,) - 最大值
        else:
            piezo_rates = last_7_days[:, :, piezo_indices]  # (B, 7)
            mean_piezometer_rate = piezo_rates.mean(dim=1)  # (B,)
            max_piezometer_rate = piezo_rates.max(dim=1)[0]  # (B,)
            
        # seismic_state (2维) - 处理多索引情况
        if isinstance(seismic_idx, list):
            seismic_rates = last_7_days[:, :, seismic_idx]  # (B, 7, num_seismic)
            mean_seismic_rate = seismic_rates.mean(dim=[1, 2])  # (B,) - 所有地震传感器的均值
            max_seismic_rate = seismic_rates.max(dim=2)[0].max(dim=1)[0]  # (B,) - 最大值
        else:
            seismic_rates = last_7_days[:, :, seismic_idx]  # (B, 7)
            mean_seismic_rate = seismic_rates.mean(dim=1)  # (B,)
            max_seismic_rate = seismic_rates.max(dim=1)[0]  # (B,)
        
        # 合并为physics_summary (B, 8)
        physics_summary = torch.stack([
            mean_velocity, mean_acceleration, mean_inverse_velocity,
            mean_rain_7d, mean_piezometer_rate, max_piezometer_rate,
            mean_seismic_rate, max_seismic_rate
        ], dim=1)  # (B, 8)
        
        return physics_summary
        
    def forward(self, h_mamba: torch.Tensor, x_dynamic_raw: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            h_mamba: (B, n_patches=11, d_model=128) - Mamba输出
            x_dynamic_raw: (B, 60, C_d) - 原始归一化后的动态输入
            
        Returns:
            h_gated: (B, 11, 128) - 门控调制后的输出
            gate_info: Dict 包含门控信息用于可解释性分析
        """
        B, n_patches, d_model = h_mamba.shape
        assert n_patches == self.n_patches, f"Expected n_patches={self.n_patches}, got {n_patches}"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"
        
        # Step 1: 物理状态提取
        physics_summary = self.extract_physics_summary(x_dynamic_raw)  # (B, 8)
        
        # Step 2: 双重门控生成
        g_dim = self.dim_gate_generator(physics_summary).unsqueeze(1)  # (B, 1, d_model)
        g_patch = self.patch_gate_generator(physics_summary).unsqueeze(-1)  # (B, n_patches, 1)
        
        # Step 3: 门控调制 (残差连接)
        # h_gated = h_mamba * (1 + g_dim * g_patch)
        gate_product = g_dim * g_patch  # (B, n_patches, d_model)
        h_gated = h_mamba * (1 + gate_product)
        
        # 准备门控信息用于可解释性
        gate_info = {
            "g_dim": g_dim,  # (B, 1, d_model)
            "g_patch": g_patch,  # (B, n_patches, 1)
            "physics_summary": physics_summary  # (B, 8)
        }
        
        return h_gated, gate_info


# 单元测试
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 创建模拟的feature_index_map
    feature_index_map = {
        'velocity_indices': 19,  # 假设GNSS_12H_velocity在第19列
        'acceleration_indices': 20,  # GNSS_12H_acceleration在第20列
        'inverse_velocity_indices': 21,  # GNSS_12H_inverse_velocity在第21列
        'piezometer_rate_indices': [22, 23, 24, 25, 26, 27],  # 6个孔压变化率
        'seismic_rate_indices': 28,  # seismic_total_rate在第28列
        'rain_7d_index': 29  # rain_7d在第29列
    }
    
    # 测试参数
    B, n_patches, d_model, T, C_d = 2, 11, 128, 60, 89
    
    # 创建随机输入
    h_mamba = torch.randn(B, n_patches, d_model)
    x_dynamic_raw = torch.randn(B, T, C_d)
    
    # 测试PhysicsGateModulator
    modulator = PhysicsGateModulator(d_model, n_patches, feature_index_map)
    h_gated, gate_info = modulator(h_mamba, x_dynamic_raw)
    
    print(f"Mamba input shape: {h_mamba.shape}")
    print(f"Dynamic input shape: {x_dynamic_raw.shape}")
    print(f"Gated output shape: {h_gated.shape}")
    print(f"Physics summary shape: {gate_info['physics_summary'].shape}")
    print(f"Dim gate shape: {gate_info['g_dim'].shape}")
    print(f"Patch gate shape: {gate_info['g_patch'].shape}")
    
    # 验证输出形状
    assert h_gated.shape == (B, n_patches, d_model), f"Output shape error: {h_gated.shape} != {(B, n_patches, d_model)}"
    assert gate_info['physics_summary'].shape == (B, 8), f"Physics summary shape error"
    assert gate_info['g_dim'].shape == (B, 1, d_model), f"Dim gate shape error"
    assert gate_info['g_patch'].shape == (B, n_patches, 1), f"Patch gate shape error"
    
    # 测试残差性质：当physics_summary全为0时，门控输出应为sigmoid(0)=0.5
    # 所以 h_gated = h_mamba * (1 + 0.5 * 0.5) = h_mamba * 1.25
    # 这里我们验证门控机制正常工作，而不是严格的残差
    x_dynamic_zero = torch.zeros_like(x_dynamic_raw)
    h_gated_zero, gate_info_zero = modulator(h_mamba, x_dynamic_zero)
    
    # 验证门控值接近0.5（因为输入为0，MLP输出接近0，sigmoid(0)=0.5）
    expected_g_dim = 0.5
    expected_g_patch = 0.5
    actual_g_dim_mean = gate_info_zero['g_dim'].mean().item()
    actual_g_patch_mean = gate_info_zero['g_patch'].mean().item()
    
    print(f"Expected g_dim: {expected_g_dim:.3f}, Actual: {actual_g_dim_mean:.3f}")
    print(f"Expected g_patch: {expected_g_patch:.3f}, Actual: {actual_g_patch_mean:.3f}")
    
    # 允许一定的误差范围
    assert abs(actual_g_dim_mean - expected_g_dim) < 0.1, f"g_dim not close to 0.5: {actual_g_dim_mean}"
    assert abs(actual_g_patch_mean - expected_g_patch) < 0.1, f"g_patch not close to 0.5: {actual_g_patch_mean}"
    
    # 测试梯度回传
    loss = h_gated.sum() + gate_info['physics_summary'].sum()
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")