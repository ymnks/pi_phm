import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
import numpy as np

class WeightedMSELoss(nn.Module):
    """加权MSE损失 - 用于增量预测"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_disp: torch.Tensor, true_disp: torch.Tensor, 
                quality_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_disp: (B, forecast_days) - 预测位移（归一化的增量值）
            true_disp: (B, forecast_days) - 真实位移（归一化的增量值）
            quality_weights: (B, T_lookback) - 质量权重
            
        Returns:
            loss: 标量损失值
        """
        B, forecast_days = pred_disp.shape
        
        # 计算每个样本的位移变化范围（基于真实值）
        disp_range = torch.max(true_disp, dim=1)[0] - torch.min(true_disp, dim=1)[0]  # (B,)
        
        # 计算batch内的中位数
        median_range = torch.median(disp_range)
        
        # 计算样本权重：高速变形样本权重最高5倍，低速样本权重约1倍
        weight_factor = (disp_range - median_range) / (median_range + 1e-6)
        sample_weights = 1 + 4 * torch.sigmoid(weight_factor)  # (B,)
        
        # 如果有质量权重，将其均值作为置信权重
        if quality_weights is not None:
            quality_confidence = torch.mean(quality_weights, dim=1)  # (B,)
            # 修复：如果所有质量权重为0，使用默认权重1.0
            quality_confidence = torch.where(
                quality_confidence < 1e-6, 
                torch.ones_like(quality_confidence), 
                quality_confidence
            )
            sample_weights = sample_weights * quality_confidence
        
        # 扩展权重到时间维度
        sample_weights_expanded = sample_weights.unsqueeze(1).expand(-1, forecast_days)  # (B, forecast_days)
        
        # 计算加权MSE
        mse = (pred_disp - true_disp) ** 2  # (B, forecast_days)
        weighted_mse = mse * sample_weights_expanded  # (B, forecast_days)
        
        return torch.mean(weighted_mse)


class AuxiliaryDisplacementLoss(nn.Module):
    """辅助位移损失"""
    
    def __init__(self, weight_coeff: float = 0.3):
        super().__init__()
        self.weight_coeff = weight_coeff
        
    def forward(self, pred_aux: torch.Tensor, true_aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_aux: (B, forecast_days, n_aux) - 预测辅助位移
            true_aux: (B, forecast_days, n_aux) - 真实辅助位移
            
        Returns:
            loss: 标量损失值
        """
        B, forecast_days, n_aux = pred_aux.shape
        
        # 检查哪些钻孔在预测窗口内全为NaN
        valid_mask = ~torch.isnan(true_aux)  # (B, forecast_days, n_aux)
        aux_valid = torch.any(valid_mask, dim=(1, 2))  # (B,) - 至少有一个有效点的样本
        
        if not torch.any(aux_valid):
            return torch.tensor(0.0, device=pred_aux.device)
            
        # 对NaN值进行处理（用0填充，但只在有效样本上计算损失）
        true_aux_clean = torch.where(torch.isnan(true_aux), torch.zeros_like(true_aux), true_aux)
        
        # 计算MSE
        mse = (pred_aux - true_aux_clean) ** 2  # (B, forecast_days, n_aux)
        
        # 只在有效样本上计算损失
        valid_mse = mse[aux_valid]  # (valid_B, forecast_days, n_aux)
        loss = torch.mean(valid_mse) * self.weight_coeff
        
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for risk classification"""
    
    def __init__(self, alpha: Optional[Union[float, List[float]]] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, n_classes) - 预测logits
            targets: (B,) - 真实标签
            
        Returns:
            loss: 标量损失值
        """
        B, n_classes = logits.shape
        
        # 确保targets是整数类型且形状正确
        if targets.dim() > 1:
            targets = targets.squeeze()
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        targets = targets.long()  # 确保是整数类型
        
        # 检查形状是否匹配
        if targets.shape[0] != B:
            raise ValueError(f"Batch size mismatch: logits has {B} samples, targets has {targets.shape[0]} samples")
        
        # 计算softmax概率
        probs = F.softmax(logits, dim=1)  # (B, n_classes)
        
        # 获取目标类别的概率
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        
        # 计算focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        ce_loss = -torch.log(target_probs + 1e-8)  # 加epsilon防止log(0)
        focal_loss = focal_weight * ce_loss
        
        # 应用alpha权重（如果提供）
        if self.alpha is not None:
            if isinstance(self.alpha, list):
                alpha_tensor = torch.tensor(self.alpha, device=logits.device)[targets]
                focal_loss = alpha_tensor * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
                
        return torch.mean(focal_loss)


class EventDetectionLoss(nn.Module):
    """蠕变爆发事件检测损失"""
    
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred_event_logits: torch.Tensor, y_event: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_event_logits: (B, 1) - 事件检测logits
            y_event: (B,) - 二分类标签 (0 or 1)
            
        Returns:
            loss: 标量损失值
        """
        # 确保y_event是浮点型（BCEWithLogitsLoss需要float标签）
        if y_event.dtype != torch.float32 and y_event.dtype != torch.float64:
            y_event = y_event.float()
            
        # 计算正样本权重（自动平衡或手动设置）
        if self.pos_weight > 0:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=pred_event_logits.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            # 自动计算正负样本比例
            num_pos = torch.sum(y_event)
            num_neg = torch.sum(1 - y_event)
            if num_pos > 0:
                auto_weight = num_neg.float() / num_pos.float()
                pos_weight_tensor = torch.tensor([auto_weight], device=pred_event_logits.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                criterion = nn.BCEWithLogitsLoss()
        
        # 计算损失
        loss = criterion(pred_event_logits.squeeze(-1), y_event)
        return loss


class CreepConstraintLoss(nn.Module):
    """蠕变约束损失（基于Table S2标注的事件期间）"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_disp: torch.Tensor, y_event: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_disp: (B, forecast_days) - 预测位移
            y_event: (B,) - 真实事件标签（可选，用于事件期间约束）
            
        Returns:
            loss: 标量损失值
        """
        B, forecast_days = pred_disp.shape
        
        if forecast_days < 3:
            return torch.tensor(0.0, device=pred_disp.device)
            
        # 计算速度（一阶差分）
        pred_vel = torch.diff(pred_disp, dim=1)  # (B, forecast_days-1)
        
        # 计算倒速率
        pred_inv_vel = 1.0 / (torch.abs(pred_vel) + 1e-8)  # (B, forecast_days-1)
        
        # 计算倒速率的变化（二阶差分）
        inv_vel_diff = torch.diff(pred_inv_vel, dim=1)  # (B, forecast_days-2)
        
        if y_event is not None:
            # 只在事件期间（y_event=1）施加约束
            is_event_period = y_event.float()  # (B,)
        else:
            # 如果没有事件标签，使用原来的加速判定逻辑
            mean_abs_vel = torch.mean(torch.abs(pred_vel), dim=1)  # (B,)
            median_vel = torch.median(mean_abs_vel)
            is_event_period = (mean_abs_vel > median_vel).float()  # (B,)
        
        # 扩展事件标志到时间维度
        is_event_expanded = is_event_period.unsqueeze(1).expand(-1, inv_vel_diff.shape[1])  # (B, forecast_days-2)
        
        # 惩罚倒速率上升的情况（inv_vel_diff > 0 表示倒速率增加，即减速）
        # 我们希望倒速率单调递减（inv_vel_diff <= 0），所以惩罚正的diff
        constraint_violation = F.relu(inv_vel_diff)  # (B, forecast_days-2)
        weighted_violation = constraint_violation * is_event_expanded
        
        return torch.mean(weighted_violation)


class HydroSeismicCausalLoss(nn.Module):
    """水压-微震时序因果约束损失"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_event_logits: torch.Tensor, x_dynamic: torch.Tensor, 
                piezometer_rate_indices: Union[int, List[int]], 
                y_event: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_event_logits: (B, 1) - 事件检测logits
            x_dynamic: (B, lookback, C_d) - 原始动态输入
            piezometer_rate_indices: 孔压变化率通道索引（单个索引或索引列表）
            y_event: (B,) - 真实事件标签（可选，用于正样本约束）
            
        Returns:
            loss: 标量损失值
        """
        B, lookback, C_d = x_dynamic.shape
        
        # 转换预测概率
        pred_event_prob = torch.sigmoid(pred_event_logits.squeeze(-1))  # (B,)
        
        # 确保索引是Python原生类型列表
        if isinstance(piezometer_rate_indices, int):
            piezo_indices_native = [piezometer_rate_indices]
        elif isinstance(piezometer_rate_indices, torch.Tensor):
            piezo_indices_native = piezometer_rate_indices.tolist()
            if isinstance(piezo_indices_native, int):
                piezo_indices_native = [piezo_indices_native]
        elif isinstance(piezometer_rate_indices, np.ndarray):
            piezo_indices_native = piezometer_rate_indices.tolist()
            if isinstance(piezo_indices_native, int):
                piezo_indices_native = [piezo_indices_native]
        elif isinstance(piezometer_rate_indices, (list, tuple)):
            piezo_indices_native = []
            for idx in piezometer_rate_indices:
                if isinstance(idx, (torch.Tensor, np.ndarray)):
                    piezo_indices_native.append(int(idx.item() if hasattr(idx, 'item') else idx))
                else:
                    piezo_indices_native.append(int(idx))
        else:
            try:
                piezo_indices_native = [int(piezometer_rate_indices)]
            except (TypeError, ValueError):
                piezo_indices_native = [0]
        
        # 提取最后3天的孔压变化率数据
        last_3_days = x_dynamic[:, -3:, :]  # (B, 3, C_d)
        piezo_rates = last_3_days[:, :, piezo_indices_native]  # (B, 3, n_piezo)
        mean_piezo_rates = torch.mean(piezo_rates, dim=2)  # (B, 3) - 平均孔压变化率
        mean_piezo_last3 = torch.mean(mean_piezo_rates, dim=1)  # (B,) - 最后3天平均孔压变化率
        
        # 计算因果约束损失
        # 当P(event) > 0.5 且 孔压在下降（mean_piezo_last3 < 0）时给予惩罚
        high_prob_mask = (pred_event_prob > 0.5).float()  # (B,)
        piezo_decreasing = F.relu(-mean_piezo_last3)  # (B,) - 孔压下降的程度
        
        causal_violation = high_prob_mask * piezo_decreasing  # (B,)
        causal_loss = torch.mean(causal_violation * pred_event_prob)  # 加权损失
        
        return causal_loss


class StressCouplingLoss(nn.Module):
    """应力耦合损失（加入Table S1季节性先验）"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_disp: torch.Tensor, x_dynamic: torch.Tensor, 
                piezometer_rate_indices: Union[int, List[int]], 
                timestamps: Optional[List[str]] = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_disp: (B, forecast_days) - 预测位移
            x_dynamic: (B, lookback, C_d) - 原始动态输入
            piezometer_rate_indices: 孔压变化率通道索引（单个索引或索引列表）
            timestamps: 预测起点的时间戳列表（用于季节性先验）
            mask: (B, lookback, C_d) - 缺失掩码
            
        Returns:
            loss: 标量损失值
        """
        B, lookback, C_d = x_dynamic.shape
        _, forecast_days = pred_disp.shape
        
        # 确保索引是Python原生类型列表
        if isinstance(piezometer_rate_indices, int):
            piezo_indices_native = [piezometer_rate_indices]
        elif isinstance(piezometer_rate_indices, torch.Tensor):
            # 如果是PyTorch张量，转换为列表
            piezo_indices_native = piezometer_rate_indices.tolist()
            if isinstance(piezo_indices_native, int):
                piezo_indices_native = [piezo_indices_native]
        elif isinstance(piezometer_rate_indices, np.ndarray):
            # 如果是NumPy数组，转换为列表
            piezo_indices_native = piezometer_rate_indices.tolist()
            if isinstance(piezo_indices_native, int):
                piezo_indices_native = [piezo_indices_native]
        elif isinstance(piezometer_rate_indices, (list, tuple)):
            # 如果是列表或元组，确保元素是原生int
            piezo_indices_native = []
            for idx in piezometer_rate_indices:
                if isinstance(idx, (torch.Tensor, np.ndarray)):
                    piezo_indices_native.append(int(idx.item() if hasattr(idx, 'item') else idx))
                else:
                    piezo_indices_native.append(int(idx))
        else:
            # 其他情况，尝试转换为单个整数
            try:
                piezo_indices_native = [int(piezometer_rate_indices)]
            except (TypeError, ValueError):
                # 如果转换失败，使用默认值
                piezo_indices_native = [0]
        
        # 提取最后7天的孔压变化率数据
        last_7_days = x_dynamic[:, -7:, :]  # (B, 7, C_d)
        
        # 检查孔压数据是否有效
        if mask is not None:
            piezo_mask = mask[:, -7:, :][:, :, piezo_indices_native]  # (B, 7, n_piezo)
            if not torch.any(piezo_mask):
                return torch.tensor(0.0, device=pred_disp.device)
        
        # 提取孔压变化率
        piezo_rates = last_7_days[:, :, piezo_indices_native]  # (B, 7, n_piezo)
        mean_piezo_rates = torch.mean(piezo_rates, dim=2)  # (B, 7) - 平均孔压变化率
        
        # 计算预测位移的速度
        pred_vel = torch.diff(pred_disp, dim=1)  # (B, forecast_days-1)
        # 为了对齐时间维度，取前6天的速度（对应最后7天的孔压）
        if pred_vel.shape[1] >= 7:
            pred_vel_aligned = pred_vel[:, :7]  # (B, 7)
        else:
            # 如果预测天数不足7天，重复最后一个速度值
            pred_vel_aligned = torch.cat([
                pred_vel, 
                pred_vel[:, -1:].repeat(1, 7 - pred_vel.shape[1])
            ], dim=1)  # (B, 7)
        
        # 计算Pearson相关系数
        # 标准化两个序列
        piezo_mean = torch.mean(mean_piezo_rates, dim=1, keepdim=True)  # (B, 1)
        piezo_std = torch.std(mean_piezo_rates, dim=1, keepdim=True) + 1e-8  # (B, 1)
        piezo_norm = (mean_piezo_rates - piezo_mean) / piezo_std  # (B, 7)
        
        vel_mean = torch.mean(pred_vel_aligned, dim=1, keepdim=True)  # (B, 1)
        vel_std = torch.std(pred_vel_aligned, dim=1, keepdim=True) + 1e-8  # (B, 1)
        vel_norm = (pred_vel_aligned - vel_mean) / vel_std  # (B, 7)
        
        # 计算相关系数
        correlation = torch.mean(piezo_norm * vel_norm, dim=1)  # (B,)
        
        # 惩罚负相关（期望正相关）
        negative_corr_penalty = F.relu(-correlation)  # (B,)
        
        # 加入季节性先验权重
        if timestamps is not None and len(timestamps) == B:
            seasonal_weights = []
            for ts in timestamps:
                try:
                    # 解析时间戳获取月份
                    if isinstance(ts, str):
                        # 尝试不同的日期格式
                        from datetime import datetime
                        dt = None
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d.%m.%Y %H:%M', '%d.%m.%Y']:
                            try:
                                dt = datetime.strptime(ts, fmt)
                                break
                            except ValueError:
                                continue
                        if dt is None:
                            month = 1  # 默认月份
                        else:
                            month = dt.month
                    else:
                        month = 1  # 默认
                    
                    # Table S1季节性先验：
                    # 6-12月（水位极大期）：孔压-位移正相关应更强（权重=1.0）
                    # 2-5月（水位极小期）：放松约束（权重=0.3）
                    # 其他月份：中等权重（0.7）
                    if 6 <= month <= 12:
                        weight = 1.0
                    elif 2 <= month <= 5:
                        weight = 0.3
                    else:
                        weight = 0.7
                    seasonal_weights.append(weight)
                except:
                    seasonal_weights.append(0.7)  # 默认权重
            
            seasonal_weights_tensor = torch.tensor(seasonal_weights, device=pred_disp.device, dtype=torch.float32)
            weighted_penalty = negative_corr_penalty * seasonal_weights_tensor
        else:
            weighted_penalty = negative_corr_penalty
        
        return torch.mean(weighted_penalty)


class SeismicCouplingLoss(nn.Module):
    """微震耦合损失"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_disp: torch.Tensor, x_dynamic: torch.Tensor, 
                seismic_rate_indices: Union[int, List[int]], mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_disp: (B, forecast_days) - 预测位移
            x_dynamic: (B, lookback, C_d) - 原始动态输入
            seismic_rate_indices: 微震率通道索引（单个索引或索引列表）
            mask: (B, lookback, C_d) - 缺失掩码
            
        Returns:
            loss: 标量损失值
        """
        B, lookback, C_d = x_dynamic.shape
        _, forecast_days = pred_disp.shape
        
        # 确保索引是Python原生类型列表
        if isinstance(seismic_rate_indices, int):
            seismic_indices_native = [seismic_rate_indices]
        elif isinstance(seismic_rate_indices, torch.Tensor):
            # 如果是PyTorch张量，转换为列表
            seismic_indices_native = seismic_rate_indices.tolist()
            if isinstance(seismic_indices_native, int):
                seismic_indices_native = [seismic_indices_native]
        elif isinstance(seismic_rate_indices, np.ndarray):
            # 如果是NumPy数组，转换为列表
            seismic_indices_native = seismic_rate_indices.tolist()
            if isinstance(seismic_indices_native, int):
                seismic_indices_native = [seismic_indices_native]
        elif isinstance(seismic_rate_indices, (list, tuple)):
            # 如果是列表或元组，确保元素是原生int
            seismic_indices_native = []
            for idx in seismic_rate_indices:
                if isinstance(idx, (torch.Tensor, np.ndarray)):
                    seismic_indices_native.append(int(idx.item() if hasattr(idx, 'item') else idx))
                else:
                    seismic_indices_native.append(int(idx))
        else:
            # 其他情况，尝试转换为单个整数
            try:
                seismic_indices_native = [int(seismic_rate_indices)]
            except (TypeError, ValueError):
                # 如果转换失败，使用默认值
                seismic_indices_native = [0]
        
        # 提取最后7天的微震率数据
        last_7_days = x_dynamic[:, -7:, :]  # (B, 7, C_d)
        seismic_rates = last_7_days[:, :, seismic_indices_native]  # (B, 7, n_seismic)
        mean_seismic_rates = torch.mean(seismic_rates, dim=2)  # (B, 7) - 平均微震率
        
        # 检查微震数据是否有效
        if mask is not None:
            seismic_mask = mask[:, -7:, :][:, :, seismic_indices_native]  # (B, 7, n_seismic)
            if not torch.any(seismic_mask):
                return torch.tensor(0.0, device=pred_disp.device)
        
        # 计算微震活动总量（7天均值）
        seis_total = torch.mean(seismic_rates, dim=1)  # (B,)
        
        # 计算预测位移总量
        total_disp = pred_disp[:, -1] - pred_disp[:, 0]  # (B,)
        
        # 标准化
        seis_mean = torch.mean(seis_total)
        seis_std = torch.std(seis_total) + 1e-8
        seis_norm = (seis_total - seis_mean) / seis_std  # (B,)
        
        disp_mean = torch.mean(total_disp)
        disp_std = torch.std(total_disp) + 1e-8
        disp_norm = (total_disp - disp_mean) / disp_std  # (B,)
        
        # 惩罚符号不一致的情况
        coupling_product = seis_norm * disp_norm  # (B,)
        penalty = F.relu(-coupling_product)  # (B,)
        
        return torch.mean(penalty)


class PIPHMLoss(nn.Module):
    """PI-PHM总损失函数"""
    
    def __init__(self, feature_index_map: Dict[str, Union[int, List[int]]], 
                 alpha_risk: float = 1.0, lambda_creep: float = 0.1, 
                 lambda_stress: float = 0.05, lambda_seismic: float = 0.05,
                 lambda_event: float = 0.5, lambda_causal: float = 0.1,
                 focal_alpha: Optional[List[float]] = None,
                 event_pos_weight: Optional[float] = None):
        super().__init__()
        self.feature_index_map = feature_index_map
        
        # 初始化各损失组件
        self.disp_loss = WeightedMSELoss()
        self.aux_disp_loss = AuxiliaryDisplacementLoss()
        # 风险分类损失 - 传递focal_alpha参数
        if focal_alpha is not None:
            self.risk_loss = FocalLoss(alpha=focal_alpha, gamma=2.0)
        else:
            self.risk_loss = FocalLoss(gamma=2.0)
        self.creep_loss = CreepConstraintLoss()
        self.stress_loss = StressCouplingLoss()
        self.seismic_loss = SeismicCouplingLoss()
        # 事件检测损失 - 使用动态pos_weight
        if event_pos_weight is not None:
            self.event_loss = EventDetectionLoss(pos_weight=event_pos_weight)
        else:
            self.event_loss = EventDetectionLoss(pos_weight=10.0)
        self.causal_loss = HydroSeismicCausalLoss()  # 新增因果约束损失
        
        # 默认权重
        self.default_weights = {
            'alpha_risk': alpha_risk,
            'lambda_creep': lambda_creep,
            'lambda_stress': lambda_stress,
            'lambda_seismic': lambda_seismic,
            'lambda_event': lambda_event,
            'lambda_causal': lambda_causal  # 新增因果约束权重
        }
        
        
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
                batch_data: Dict[str, torch.Tensor], 
                loss_weights: Optional[Dict[str, float]] = None) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_outputs: 模型输出字典
            batch_data: 批次数据字典，包含：
                - y_disp_main: (B, forecast_days)
                - y_disp_aux: (B, forecast_days, n_aux)
                - y_risk: (B,)
                - y_event: (B,)  # 新增事件标签
                - x_dynamic: (B, lookback, C_d)
                - mask: (B, lookback, C_d)
                - quality_weights: (B, lookback)
                - timestamp: List[str]  # 时间戳（用于季节性先验）
            loss_weights: 当前epoch的损失权重字典
            
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        # 使用提供的权重或默认权重
        weights = loss_weights if loss_weights is not None else self.default_weights
        
        # 主位移损失（仅当模型输出包含pred_disp时）
        if 'pred_disp' in model_outputs:
            disp_loss = self.disp_loss(
                model_outputs['pred_disp'], 
                batch_data['y_disp_main'],
                batch_data.get('quality_weights', None)
                # 移除normalizer参数，因为现在标签和预测都在同一归一化空间
            )
        else:
            disp_loss = torch.tensor(0.0, device=batch_data['y_disp_main'].device)
        
        # 辅助位移损失（仅当模型输出包含pred_aux_disp时）
        if 'pred_aux_disp' in model_outputs:
            aux_loss = self.aux_disp_loss(
                model_outputs['pred_aux_disp'],
                batch_data['y_disp_aux']
            )
        else:
            aux_loss = torch.tensor(0.0, device=disp_loss.device)
        
        # 风险分类损失（仅当模型输出包含pred_risk_logits时）
        if 'pred_risk_logits' in model_outputs:
            risk_loss = self.risk_loss(
                model_outputs['pred_risk_logits'],
                batch_data['y_risk']
            )
        else:
            risk_loss = torch.tensor(0.0, device=disp_loss.device)
        
        # 物理约束损失（仅当模型输出包含pred_disp时）
        if 'pred_disp' in model_outputs:
            creep_constr = self.creep_loss(
                model_outputs['pred_disp'],
                batch_data.get('y_event', None)  # 传递事件标签用于事件期间约束
            )
            
            stress_constr = self.stress_loss(
                model_outputs['pred_disp'],
                batch_data['x_dynamic'],
                self.feature_index_map['piezometer_rate_indices'],
                batch_data.get('timestamp', None),  # 传递时间戳用于季节性先验
                batch_data.get('mask', None)
            )
            
            seismic_constr = self.seismic_loss(
                model_outputs['pred_disp'],
                batch_data['x_dynamic'],
                self.feature_index_map['seismic_rate_indices'],
                batch_data.get('mask', None)
            )
        else:
            # 如果没有位移预测，物理约束损失为0
            creep_constr = torch.tensor(0.0, device=batch_data['y_disp_main'].device)
            stress_constr = torch.tensor(0.0, device=batch_data['y_disp_main'].device)
            seismic_constr = torch.tensor(0.0, device=batch_data['y_disp_main'].device)
        
        # 新增因果约束损失（仅当模型输出包含pred_event_logits时）
        if 'pred_event_logits' in model_outputs:
            causal_constr = self.causal_loss(
                model_outputs['pred_event_logits'],
                batch_data['x_dynamic'],
                self.feature_index_map['piezometer_rate_indices'],
                batch_data.get('y_event', None)
            )
        else:
            causal_constr = torch.tensor(0.0, device=disp_loss.device)
        
        # 事件检测损失（新增）
        if 'y_event' in batch_data and 'pred_event_logits' in model_outputs:
            event_loss = self.event_loss(
                model_outputs['pred_event_logits'],
                batch_data['y_event']
            )
        else:
            event_loss = torch.tensor(0.0, device=disp_loss.device)
        
        # 计算总损失
        total_loss = (
            disp_loss + aux_loss + 
            weights['alpha_risk'] * risk_loss +
            weights['lambda_event'] * event_loss +
            weights['lambda_creep'] * creep_constr +
            weights['lambda_stress'] * stress_constr +
            weights['lambda_seismic'] * seismic_constr +
            weights['lambda_causal'] * causal_constr  # 新增因果约束项
        )
        
        loss_dict = {
            'disp_loss': disp_loss.item(),
            'aux_loss': aux_loss.item(),
            'risk_loss': risk_loss.item(),
            'event_loss': event_loss.item(),
            'creep_constr': creep_constr.item(),
            'stress_constr': stress_constr.item(),
            'seismic_constr': seismic_constr.item(),
            'causal_constr': causal_constr.item(),  # 新增因果约束项
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# 单元测试
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 测试参数
    B, lookback, C_d, forecast_days, n_aux = 4, 60, 89, 7, 6
    
    # 创建模拟数据（需要梯度）
    pred_disp = torch.randn(B, forecast_days, requires_grad=True)
    true_disp = torch.randn(B, forecast_days)
    pred_aux = torch.randn(B, forecast_days, n_aux, requires_grad=True)
    true_aux = torch.randn(B, forecast_days, n_aux)
    logits = torch.randn(B, 4, requires_grad=True)
    targets = torch.randint(0, 4, (B,))
    x_dynamic = torch.randn(B, lookback, C_d, requires_grad=True)
    mask = torch.ones(B, lookback, C_d, dtype=torch.bool)
    quality_weights = torch.ones(B, lookback)
    
    # 创建feature_index_map
    feature_index_map = {
        'velocity_indices': 19,
        'acceleration_indices': 20,
        'inverse_velocity_indices': 21,
        'piezometer_rate_indices': [22, 23, 24, 25, 26, 27],
        'seismic_rate_indices': 28,
        'rain_7d_index': 29
    }
    
    # 测试各损失函数
    print("Testing WeightedMSELoss...")
    disp_loss_fn = WeightedMSELoss()
    loss1 = disp_loss_fn(pred_disp, true_disp, quality_weights)
    assert loss1 >= 0, f"WeightedMSELoss should be >= 0, got {loss1}"
    loss1.backward(retain_graph=True)
    print(f"  Loss: {loss1.item():.4f}, Gradient exists")
    
    print("Testing AuxiliaryDisplacementLoss...")
    aux_loss_fn = AuxiliaryDisplacementLoss()
    loss2 = aux_loss_fn(pred_aux, true_aux)
    assert loss2 >= 0, f"AuxiliaryDisplacementLoss should be >= 0, got {loss2}"
    loss2.backward(retain_graph=True)
    print(f"  Loss: {loss2.item():.4f}, Gradient exists")
    
    print("Testing FocalLoss...")
    risk_loss_fn = FocalLoss(gamma=2.0)
    loss3 = risk_loss_fn(logits, targets)
    assert loss3 >= 0, f"FocalLoss should be >= 0, got {loss3}"
    loss3.backward(retain_graph=True)
    print(f"  Loss: {loss3.item():.4f}, Gradient exists")
    
    print("Testing CreepConstraintLoss...")
    creep_loss_fn = CreepConstraintLoss()
    loss4 = creep_loss_fn(pred_disp)
    assert loss4 >= 0, f"CreepConstraintLoss should be >= 0, got {loss4}"
    loss4.backward(retain_graph=True)
    print(f"  Loss: {loss4.item():.4f}, Gradient exists")
    
    print("Testing StressCouplingLoss...")
    stress_loss_fn = StressCouplingLoss()
    loss5 = stress_loss_fn(pred_disp, x_dynamic, feature_index_map['piezometer_rate_indices'], mask)
    assert loss5 >= 0, f"StressCouplingLoss should be >= 0, got {loss5}"
    loss5.backward(retain_graph=True)
    print(f"  Loss: {loss5.item():.4f}, Gradient exists")
    
    print("Testing SeismicCouplingLoss...")
    seismic_loss_fn = SeismicCouplingLoss()
    loss6 = seismic_loss_fn(pred_disp, x_dynamic, feature_index_map['seismic_rate_indices'], mask)
    assert loss6 >= 0, f"SeismicCouplingLoss should be >= 0, got {loss6}"
    loss6.backward(retain_graph=True)
    print(f"  Loss: {loss6.item():.4f}, Gradient exists")
    
    print("Testing PIPHMLoss...")
    piphm_loss_fn = PIPHMLoss(feature_index_map)
    model_outputs = {
        'pred_disp': pred_disp,
        'pred_aux_disp': pred_aux,
        'pred_risk_logits': logits
    }
    batch_data = {
        'y_disp_main': true_disp,
        'y_disp_aux': true_aux,
        'y_risk': targets,
        'x_dynamic': x_dynamic,
        'mask': mask,
        'quality_weights': quality_weights
    }
    total_loss, loss_dict = piphm_loss_fn(model_outputs, batch_data)
    assert total_loss >= 0, f"PIPHMLoss should be >= 0, got {total_loss}"
    total_loss.backward()
    print(f"  Total Loss: {total_loss.item():.4f}, All components: {list(loss_dict.keys())}")
    
    print("All tests passed!")