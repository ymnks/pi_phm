"""数据集模块 - 包含物理感知归一化器和时序数据集类"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import pickle
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsAwareNormalizer:
    """物理感知归一化器 - 支持绝对位移和增量位移归一化"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        self.config = config
        self.original_series = ['GNSS_12H', 'KH0206_Displacement', 'KH0112_Displacement', 
                               'KH0117_Displacement', 'KH0118_Displacement', 'KH0217_Displacement',
                               'KH0218_Displacement', 'KH0306_Displacement']
        self.feature_types = {}
        self.scalers = {}
        self.original_columns = []
        self.increment_scalers = {}  # 专门用于增量值的scaler
        
    def fit(self, df_features: pd.DataFrame, feature_names: Optional[List[str]] = None, train_mask: Optional[np.ndarray] = None):
        """在训练集上拟合归一化参数（包括绝对位移和增量位移）"""
        logger.info("开始拟合物理感知归一化器...")
        self.original_columns = list(df_features.columns)
        
        # 如果提供了train_mask，则使用它来获取训练数据，否则使用整个df_features
        if train_mask is not None:
            train_data = df_features[train_mask]
        else:
            train_data = df_features
        
        # 确定特征类型并拟合绝对位移scaler
        for col in df_features.columns:
            if col in self.original_series:
                self.feature_types[col] = 'A'
                self.scalers[col] = StandardScaler()
                # 只对非NaN值进行拟合
                valid_data = train_data[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    self.scalers[col].fit(valid_data)
            elif any(keyword in col.lower() for keyword in ['velocity', 'acceleration', 'jerk', 'rate']):
                self.feature_types[col] = 'B'
                self.scalers[col] = RobustScaler()
                valid_data = train_data[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    self.scalers[col].fit(valid_data)
            else:
                self.feature_types[col] = 'C'
                # 类型C不使用scaler，只做clip
        
        # 为增量位移拟合专门的scaler
        target_col = self.config.data.target_col
        if target_col in train_data.columns:
            # 计算训练集上的增量值
            target_series = train_data[target_col].dropna()
            increments = target_series.diff(7).dropna()  # 7天增量
            
            if len(increments) > 0:
                self.increment_scalers[target_col] = StandardScaler()
                increment_data = increments.values.reshape(-1, 1)
                self.increment_scalers[target_col].fit(increment_data)
                
                # 也为辅助目标列拟合增量scaler
                for aux_col in self.config.data.auxiliary_targets:
                    if aux_col in train_data.columns:
                        aux_series = train_data[aux_col].dropna()
                        aux_increments = aux_series.diff(7).dropna()
                        if len(aux_increments) > 0:
                            self.increment_scalers[aux_col] = StandardScaler()
                            aux_increment_data = aux_increments.values.reshape(-1, 1)
                            self.increment_scalers[aux_col].fit(aux_increment_data)
        
        logger.info(f"归一化器拟合完成! 特征类型分布: A={sum(1 for t in self.feature_types.values() if t=='A')}, B={sum(1 for t in self.feature_types.values() if t=='B')}, C={sum(1 for t in self.feature_types.values() if t=='C')}")
        logger.info(f"增量归一化器数量: {len(self.increment_scalers)}")
    
    def transform_increment(self, increment_values: np.ndarray, column_name: str) -> np.ndarray:
        """对增量值进行归一化"""
        if column_name in self.increment_scalers:
            return self.increment_scalers[column_name].transform(increment_values.reshape(-1, 1)).flatten()
        else:
            # 如果没有增量scaler，返回原始值
            return increment_values
    
    def inverse_transform_increment(self, normalized_increments: np.ndarray, column_name: str) -> np.ndarray:
        """对归一化的增量值进行反归一化"""
        if column_name in self.increment_scalers:
            return self.increment_scalers[column_name].inverse_transform(normalized_increments.reshape(-1, 1)).flatten()
        else:
            # 如果没有增量scaler，返回原始值
            return normalized_increments
    
    def transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """应用归一化到特征矩阵"""
        df_normalized = df_features.copy()
        
        for col in df_features.columns:
            if col not in self.feature_types:
                continue
                
            feature_type = self.feature_types[col]
            series = df_features[col].copy()
            
            if feature_type == 'A' or feature_type == 'B':
                # 使用对应的scaler
                if col in self.scalers and hasattr(self.scalers[col], 'scale_'):
                    # 只对非NaN值进行变换
                    valid_mask = series.notna()
                    if valid_mask.any():
                        valid_values = series[valid_mask].values.reshape(-1, 1)
                        normalized_values = self.scalers[col].transform(valid_values).flatten()
                        series[valid_mask] = normalized_values
                df_normalized[col] = series
            
            elif feature_type == 'C':
                # 只做clip到[-5, 5]
                series = series.clip(lower=-5, upper=5)
                df_normalized[col] = series
        
        return df_normalized
    
    def inverse_transform(self, df_features: pd.DataFrame, target_col: str = "GNSS_12H") -> pd.DataFrame:
        """
        反归一化特征矩阵
        
        Args:
            df_features: 归一化后的特征矩阵
            target_col: 主目标列名
            
        Returns:
            pd.DataFrame: 反归一化后的特征矩阵
        """
        df_denormalized = df_features.copy()
        
        for col in df_features.columns:
            if col not in self.feature_types:
                continue
                
            feature_type = self.feature_types[col]
            series = df_features[col].copy()
            
            if feature_type == 'A' or feature_type == 'B':
                if col in self.scalers and hasattr(self.scalers[col], 'scale_'):
                    valid_mask = series.notna()
                    if valid_mask.any():
                        valid_values = series[valid_mask].values.reshape(-1, 1)
                        denormalized_values = self.scalers[col].inverse_transform(valid_values).flatten()
                        series[valid_mask] = denormalized_values
                df_denormalized[col] = series
            # 类型C不需要反归一化
        
        return df_denormalized
    
    def inverse_transform_single_column(self, values: np.ndarray, column_idx: int = 0) -> np.ndarray:
        """
        反归一化单列数据
        
        Args:
            values: 需要反归一化的数组 (shape: [...])
            column_idx: 列索引，默认为0（对应GNSS_12H）
            
        Returns:
            np.ndarray: 反归一化后的数组
        """
        if self.original_columns is None or len(self.original_columns) <= column_idx:
            # 如果没有原始列信息或索引越界，直接返回原值
            return values
            
        target_col = self.original_columns[column_idx]
        
        if target_col not in self.feature_types:
            return values
            
        feature_type = self.feature_types[target_col]
        
        if feature_type == 'A' or feature_type == 'B':
            if target_col in self.scalers and hasattr(self.scalers[target_col], 'scale_'):
                # 重塑为2D数组进行反归一化
                original_shape = values.shape
                values_2d = values.reshape(-1, 1)
                denormalized_values = self.scalers[target_col].inverse_transform(values_2d)
                return denormalized_values.reshape(original_shape)
        
        # 类型C不需要反归一化
        return values
    
    def save(self, filepath: str):
        """保存归一化器参数"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_types': self.feature_types,
                'original_columns': self.original_columns
            }, f)
        logger.info(f"归一化器参数已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载归一化器参数"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.feature_types = data['feature_types']
            self.original_columns = data['original_columns']
        logger.info(f"归一化器参数已从 {filepath} 加载")


def generate_event_detection_labels(
    event_catalog: 'CreepBurstCatalog',
    df_features: pd.DataFrame,
    forecast_horizon: int = 7
) -> pd.Series:
    """
    生成蠕变爆发事件检测标签
    
    Args:
        event_catalog: 蠕变爆发事件目录
        df_features: 特征DataFrame，index为日期
        forecast_horizon: 预测时间窗口（天数）
        
    Returns:
        y_event: 二分类标签Series (0 or 1)
    """
    logger.info(f"生成事件检测标签 (预测窗口: {forecast_horizon}天)...")
    
    # 初始化标签为0
    y_event = pd.Series(0, index=df_features.index, dtype=int)
    
    # 获取所有事件的开始时间
    event_start_times = [event.start_time for event in event_catalog.events]
    
    # 对每个时间点，检查未来forecast_horizon天内是否有事件开始
    for date in df_features.index:
        # 计算预测窗口的结束时间
        window_end = date + pd.Timedelta(days=forecast_horizon)
        
        # 检查是否有事件在[date, window_end]期间开始
        has_event = any(
            (event_start >= date) and (event_start <= window_end)
            for event_start in event_start_times
        )
        
        if has_event:
            y_event[date] = 1
    
    # 输出统计信息
    total_days = len(y_event)
    positive_days = y_event.sum()
    negative_days = total_days - positive_days
    positive_ratio = positive_days / total_days if total_days > 0 else 0
    
    logger.info(f"事件检测标签统计:")
    logger.info(f"  总天数: {total_days}")
    logger.info(f"  正样本 (y_event=1): {positive_days}天 ({positive_ratio:.1%})")
    logger.info(f"  负样本 (y_event=0): {negative_days}天 ({1-positive_ratio:.1%})")
    
    return y_event


class AknesTimeSeriesDataset(Dataset):
    """Åknes滑坡时序数据集"""
    
    def __init__(self, df_features: pd.DataFrame, df_quality: Optional[pd.DataFrame], 
                 labels: pd.Series, static_features: np.ndarray, config: 'PI_PHM_Config',
                 mode: str = 'train', y_event: Optional[pd.Series] = None,
                 normalizer: Optional['PhysicsAwareNormalizer'] = None):  # 添加normalizer参数
        """
        Args:
            df_features: 特征DataFrame (n_samples, n_features)
            df_quality: 质量标记DataFrame (n_samples, n_features) - quality_flag: 0=原始, 1=插值高质量, 2=插值低质量, 3=仍缺失
            labels: 风险标签Series (n_samples,)
            static_features: 静态地质特征 (n_static_features,)
            config: 配置对象
            mode: 'train', 'val', 'test'
            y_event: 事件检测标签Series (n_samples,) - 可选
            normalizer: PhysicsAwareNormalizer - 可选，用于归一化标签
        """
        self.df_features = df_features
        self.df_quality = df_quality
        self.labels = labels
        self.static_features = static_features
        self.config = config
        self.mode = mode
        self.y_event = y_event
        self.normalizer = normalizer  # 保存normalizer
        
        # 配置参数
        self.lookback = config.model.lookback
        self.forecast = config.model.forecast
        self.auxiliary_target_cols = config.data.auxiliary_targets
        
        # 找到所有有效样本索引
        self.valid_indices = self._find_valid_indices()
        
        # 计算样本权重（任务6：事件检测正样本15倍权重）
        if mode == 'train' and len(self.valid_indices) > 0:
            if self.y_event is not None:
                # 使用事件检测标签计算权重
                self.sample_weights = compute_sample_weights_with_event(
                    labels, self.y_event, self.valid_indices, self.lookback, self.forecast, config
                )
            else:
                # 回退到原始风险标签权重
                self.sample_weights = compute_sample_weights(
                    labels, self.valid_indices, self.lookback, self.forecast, config
                )
            print(f"Sample weights created successfully: {len(self.sample_weights)} samples")
        else:
            self.sample_weights = np.ones(len(self.valid_indices))
    
    def _find_valid_indices(self) -> List[int]:
        """找到所有有效样本索引"""
        valid_indices = []
        n_samples = len(self.df_features) - self.lookback - self.forecast + 1
        
        # 调试：如果数据集太小，直接返回空列表
        if n_samples <= 0:
            logger.warning(f"数据集长度不足: {len(self.df_features)}, 需要至少 {self.lookback + self.forecast} 个样本")
            return valid_indices
        
        # 定义关键特征（用于缺失率计算），排除已知高缺失率的孔压特征
        key_features = [
            'GNSS_12H', 'KH0206_Displacement', 'KH0112_Displacement', 
            'KH0117_Displacement', 'KH0118_Displacement', 'KH0218_Displacement', 'KH0306_Displacement',
            'Seismicity_borehole', 'Seismicity_surface',
            'Air_temperature', 'Precipitation', 'Snow', 'Surface_runoff'
        ]
        # 只保留实际存在于数据中的关键特征
        actual_key_features = [col for col in key_features if col in self.df_features.columns]
        
        if not actual_key_features:
            logger.warning("未找到任何关键特征，使用所有特征进行缺失率计算")
            actual_key_features = list(self.df_features.columns)
        
        logger.debug(f"使用 {len(actual_key_features)} 个关键特征进行缺失率计算: {actual_key_features[:5]}...")
        
        for i in range(n_samples):
            # 输入窗口: [i, i+lookback)
            # 预测窗口: [i+lookback, i+lookback+forecast)
            
            input_window = slice(i, i + self.lookback)
            forecast_window = slice(i + self.lookback, i + self.lookback + self.forecast)
            
            # 检查输入窗口的关键特征平均缺失率（使用50%阈值）
            input_features = self.df_features.iloc[input_window][actual_key_features]
            input_missing_rate = input_features.isna().mean().mean()
            if input_missing_rate >= 0.5:  # 使用50%阈值
                continue
            
            # 检查预测窗口主目标的有效数据（至少1天）
            if self.config.data.target_col in self.df_features.columns:
                target_forecast = self.df_features[self.config.data.target_col].iloc[forecast_window]
                valid_target_days = target_forecast.notna().sum()
                if valid_target_days < 1:
                    continue
            else:
                # 如果目标列不存在，跳过此样本
                logger.warning(f"目标列 {self.config.data.target_col} 不存在于数据中")
                continue
            
            valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        start_idx = self.valid_indices[idx]
        end_input = start_idx + self.lookback
        end_forecast = end_input + self.forecast
        
        # 动态特征
        x_dynamic = self.df_features.iloc[start_idx:end_input].values
        # 确保x_dynamic是2D
        if len(x_dynamic.shape) == 1:
            # 如果是1D，添加时间维度
            x_dynamic = x_dynamic.reshape(1, -1)
        # 处理NaN值
        x_dynamic = np.nan_to_num(x_dynamic, nan=0.0)
        x_dynamic = torch.FloatTensor(x_dynamic)
        
        # 静态特征
        x_static = torch.FloatTensor(self.static_features)
        
        # 掩码 (True=真实值, False=缺失/插值)
        # 处理维度不匹配问题：如果df_quality列数少于df_features，需要扩展
        if self.df_quality is not None:
            quality_window_raw = self.df_quality.iloc[start_idx:end_input].values
            
            if quality_window_raw.shape[1] < x_dynamic.shape[1]:
                # 扩展质量矩阵以匹配特征数量
                # 原始特征数量
                original_features = quality_window_raw.shape[1]
                # 目标特征数量
                target_features = x_dynamic.shape[1]
                
                # 创建扩展的质量窗口
                quality_window = np.ones((quality_window_raw.shape[0], target_features))
                
                # 复制原始质量信息
                quality_window[:, :original_features] = quality_window_raw
                
                # 对于衍生特征，假设它们的质量与最后一个原始特征相同
                if original_features > 0:
                    last_original_quality = quality_window_raw[:, -1:]
                    # 将最后一个原始特征的质量复制到所有衍生特征
                    quality_window[:, original_features:] = np.tile(last_original_quality, (1, target_features - original_features))
                else:
                    # 如果没有原始特征，所有衍生特征都标记为高质量（1.0）
                    quality_window[:, original_features:] = 1.0
            else:
                quality_window = quality_window_raw
            
            mask = (quality_window != 3)  # quality_flag=3表示仍缺失
            mask = torch.BoolTensor(mask)
            
            # 移除维度检查断言，因为已经处理了维度匹配
            
            # 质量权重
            quality_weights = np.ones(self.lookback)
            # quality_window 是 (lookback, C_d) 的2D数组
            # 我们需要为每个时间步计算一个权重，基于该时间步所有特征的质量标记
            for t in range(self.lookback):
                time_step_quality = quality_window[t]
                # 如果该时间步有任何特征是仍缺失（3），则权重为0
                if 3 in time_step_quality:
                    quality_weights[t] = 0.0
                # 否则取最差的质量等级来决定权重
                elif 2 in time_step_quality:
                    quality_weights[t] = 0.5
                elif 1 in time_step_quality:
                    quality_weights[t] = 0.8
                else:
                    quality_weights[t] = 1.0
            quality_weights = torch.FloatTensor(quality_weights)
        else:
            # 如果没有质量数据，默认全为有效且权重为1
            mask = torch.ones((self.lookback, x_dynamic.shape[1]), dtype=torch.bool)
            quality_weights = torch.ones(self.lookback)
        
        # 主目标 - 使用归一化后的位移增量值（相对于最后一个输入时间点）
        y_disp_main = np.zeros(self.forecast)
        if self.config.data.target_col in self.df_features.columns:
            # 获取最后一个输入时间点的值（用于计算增量）
            last_input_idx = end_input - 1
            if last_input_idx >= 0:
                last_input_value = self.df_features[self.config.data.target_col].iloc[last_input_idx]
            else:
                last_input_value = 0.0
            
            # 获取预测窗口的GNSS值（绝对位移）
            forecast_values = self.df_features[self.config.data.target_col].iloc[end_input:end_forecast].values
            
            # 计算相对于最后一个输入时间点的增量
            increments = np.zeros_like(forecast_values)
            for i in range(len(forecast_values)):
                if pd.isna(forecast_values[i]) or pd.isna(last_input_value):
                    increments[i] = 0.0  # NaN用0填充
                else:
                    increments[i] = forecast_values[i] - last_input_value
            
            # 对增量进行归一化
            if self.normalizer is not None:
                normalized_increments = self.normalizer.transform_increment(increments, self.config.data.target_col)
                y_disp_main = normalized_increments
            else:
                y_disp_main = increments
        
        y_disp_main = np.nan_to_num(y_disp_main, nan=0.0)
        # 确保y_disp_main是一维向量，形状为(forecast,)而非(forecast, 1)或(1, forecast)
        if len(y_disp_main.shape) > 1:
            logger.warning(f"y_disp_main has unexpected shape {y_disp_main.shape}, flattening...")
            y_disp_main = y_disp_main.flatten()
        y_disp_main = torch.FloatTensor(y_disp_main)
        
        # 【关键修复】验证y_disp_main的形状
        if len(y_disp_main.shape) != 1:
            raise ValueError(f"y_disp_main must be 1D with shape (forecast,), got {y_disp_main.shape}")
        if y_disp_main.shape[0] != self.forecast:
            raise ValueError(f"y_disp_main length {y_disp_main.shape[0]} != forecast {self.forecast}")
        
        # 辅助目标 - 也使用归一化后的位移增量值
        y_disp_aux = np.zeros((self.forecast, len(self.auxiliary_target_cols)))
        
        # 调试：检查哪些辅助目标列存在
        existing_aux_cols = [col for col in self.auxiliary_target_cols if col in self.df_features.columns]
        if len(existing_aux_cols) != len(self.auxiliary_target_cols):
            if not hasattr(self, '_printed_aux_warning'):
                logger.warning(f"Only {len(existing_aux_cols)}/{len(self.auxiliary_target_cols)} auxiliary targets available: {existing_aux_cols}")
                self._printed_aux_warning = True
        
        for j, aux_col in enumerate(self.auxiliary_target_cols):
            if aux_col in self.df_features.columns:
                # 获取最后一个输入时间点的辅助值
                last_input_idx = end_input - 1
                if last_input_idx >= 0:
                    last_input_aux_value = self.df_features[aux_col].iloc[last_input_idx]
                else:
                    last_input_aux_value = 0.0
                
                # 获取预测窗口的辅助值
                aux_forecast_values = self.df_features[aux_col].iloc[end_input:end_forecast].values
                
                # 计算增量
                aux_increments = np.zeros_like(aux_forecast_values)
                for i in range(len(aux_forecast_values)):
                    if pd.isna(aux_forecast_values[i]) or pd.isna(last_input_aux_value):
                        aux_increments[i] = 0.0
                    else:
                        aux_increments[i] = aux_forecast_values[i] - last_input_aux_value
                
                # 对增量进行归一化
                if self.normalizer is not None:
                    normalized_aux_increments = self.normalizer.transform_increment(aux_increments, aux_col)
                    y_disp_aux[:, j] = normalized_aux_increments
                else:
                    y_disp_aux[:, j] = aux_increments
        y_disp_aux = np.nan_to_num(y_disp_aux, nan=0.0)
        y_disp_aux = torch.FloatTensor(y_disp_aux)
        
        # 风险标签（未来7天内的最高风险等级）
        future_labels = self.labels.iloc[end_input:end_forecast]
        if len(future_labels) > 0:
            y_risk = int(future_labels.max())  # 直接使用max()返回的标量值
        else:
            y_risk = 0
        y_risk = torch.LongTensor([y_risk])
        
        # 事件检测标签（新增）
        if self.y_event is not None:
            # 事件标签对应输入窗口的最后一天
            y_event_label = int(self.y_event.iloc[start_idx + self.lookback - 1])
        else:
            y_event_label = 0
        
        # 调试：打印target shape（步骤1.2）
        if not hasattr(self, '_printed'):
            print(f"y_disp_main shape: {y_disp_main.shape}")
            print(f"y_disp_main values: {y_disp_main}")
            self._printed = True
        
        return {
            'x_dynamic': x_dynamic,
            'x_static': x_static,
            'mask': mask,
            'quality_weights': quality_weights,
            'y_disp_main': y_disp_main,
            'y_disp_aux': y_disp_aux,
            'y_risk': y_risk,
            'y_event': y_event_label,  # 新增事件检测标签
            'timestamp': str(self.df_features.index[start_idx + self.lookback - 1])
        }
    
    def _apply_data_augmentation(self, x_dynamic, y_disp_main, y_disp_aux):
        """应用数据增强"""
        import random
        
        # 30%概率添加高斯噪声
        if random.random() < 0.3:
            # 获取各通道的标准差（从归一化器或数据中估计）
            noise_scale = 0.01
            noise = torch.randn_like(x_dynamic) * noise_scale
            x_dynamic = x_dynamic + noise
        
        # 20%概率随机mask 1-5天连续时间步
        if random.random() < 0.2:
            mask_days = random.randint(1, min(5, self.lookback))
            start_day = random.randint(0, self.lookback - mask_days)
            x_dynamic[start_day:start_day + mask_days] = 0.0
        
        return x_dynamic, y_disp_main, y_disp_aux


def create_static_features(config: object) -> np.ndarray:
    """构建静态地质特征向量"""
    # 从config中获取地质参数
    slope_angle = getattr(config.model, 'avg_slope', 35.0)
    joint_dip_direction = getattr(config.model, 'joint_dip_direction', 125.0)
    joint_dip_angle = getattr(config.model, 'joint_dip_angle', 35.0)
    cohesion = getattr(config.model, 'cohesion', 50.0)
    friction_angle = getattr(config.model, 'friction_angle', 31.0)
    density = getattr(config.model, 'rock_density', 2700.0)
    
    # 构建6维静态特征向量
    static_features = np.array([
        slope_angle / 90.0,                           # 归一化坡度
        np.sin(joint_dip_direction * np.pi / 180),   # 节理倾向正弦
        joint_dip_angle / 90.0,                      # 归一化节理倾角
        cohesion / 200.0,                            # 归一化粘聚力
        friction_angle / 45.0,                       # 归一化摩擦角
        density / 3000.0                             # 归一化密度
    ])
    
    return static_features


def compute_sample_weights(labels: pd.Series, valid_indices: List[int], 
                          lookback: int, forecast: int, 
                          config: Optional['PI_PHM_Config'] = None) -> np.ndarray:
    """计算样本权重用于WeightedRandomSampler"""
    # 获取每个样本的标签（未来7天内最高风险等级）
    sample_labels = []
    sample_timestamps = []  # 存储每个样本的时间戳
    
    for idx in valid_indices:
        end_input = idx + lookback
        end_forecast = end_input + forecast
        future_labels = labels.iloc[end_input:end_forecast]
        if len(future_labels) > 0:
            max_label_val = future_labels.max()
            # 处理pandas Series返回值，确保获取标量
            if hasattr(max_label_val, 'item'):
                max_label = int(max_label_val.item())
            else:
                max_label = int(max_label_val)
        else:
            max_label = 0
        sample_labels.append(int(max_label))  # 确保添加的是Python原生int
        
        # 记录样本时间戳（输入窗口的最后一天）
        sample_timestamps.append(labels.index[end_input - 1])
    
    sample_labels = np.array(sample_labels)
    sample_timestamps = pd.DatetimeIndex(sample_timestamps)
    
    # 计算基础类别权重（基于逆频率）
    unique_labels, counts = np.unique(sample_labels, return_counts=True)
    # 转换为Python原生类型以避免unhashable type错误
    unique_labels = [int(l) for l in unique_labels]
    counts = [float(c) for c in counts]  # 确保counts也是Python原生类型
    label_weights = dict(zip(unique_labels, [1.0 / c for c in counts]))
    
    # 为每个样本分配基础权重（确保label是Python原生类型）
    sample_weights = np.array([label_weights[int(label)] for label in sample_labels])
    
    # 应用事件感知加权（如果提供了配置）
    if config is not None and hasattr(config.data, 'acceleration_events'):
        known_events = getattr(config.data, 'acceleration_events', [])
        if known_events:
            # 创建事件期间的掩码
            event_mask = np.zeros(len(sample_weights), dtype=bool)
            for event_start, event_end in known_events:
                start_dt = pd.to_datetime(event_start)
                end_dt = pd.to_datetime(event_end)
                event_period_mask = (sample_timestamps >= start_dt) & (sample_timestamps <= end_dt)
                event_mask |= event_period_mask
            
            # 对事件期间的样本应用10倍权重
            sample_weights[event_mask] *= 10.0
            
            # 对其他YELLOW/RED样本应用5倍权重
            yellow_red_mask = (sample_labels >= 2) & (~event_mask)
            sample_weights[yellow_red_mask] *= 5.0
            # GREEN/BLUE样本保持原权重（乘以1.0）
    
    # 归一化权重
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    return sample_weights


def compute_sample_weights_with_event(labels: pd.Series, y_event: pd.Series, 
                                   valid_indices: List[int], lookback: int, forecast: int,
                                   config: Optional['PI_PHM_Config'] = None) -> np.ndarray:
    """计算样本权重，优先使用事件检测标签（任务6：正样本15倍权重）"""
    # 获取每个样本的事件检测标签（输入窗口最后一天的y_event值）
    sample_event_labels = []
    sample_risk_labels = []
    sample_timestamps = []
    
    for idx in valid_indices:
        # 事件检测标签对应输入窗口的最后一天
        event_label = int(y_event.iloc[idx + lookback - 1])
        sample_event_labels.append(event_label)
        
        # 风险标签（未来7天内的最高风险等级）
        end_input = idx + lookback
        end_forecast = end_input + forecast
        future_labels = labels.iloc[end_input:end_forecast]
        if len(future_labels) > 0:
            max_label_val = future_labels.max()
            if hasattr(max_label_val, 'item'):
                max_label = int(max_label_val.item())
            else:
                max_label = int(max_label_val)
        else:
            max_label = 0
        sample_risk_labels.append(int(max_label))
        
        # 记录样本时间戳
        sample_timestamps.append(labels.index[end_input - 1])
    
    sample_event_labels = np.array(sample_event_labels)
    sample_risk_labels = np.array(sample_risk_labels)
    sample_timestamps = pd.DatetimeIndex(sample_timestamps)
    
    # 基础权重：所有样本初始权重为1.0
    sample_weights = np.ones(len(sample_event_labels))
    
    # 任务6要求：正样本（事件发生或前驱窗口内）权重15倍
    # 因为正样本约占10-15%，15倍权重近似平衡
    positive_mask = (sample_event_labels == 1)
    sample_weights[positive_mask] *= 15.0
    
    # 对YELLOW/RED风险样本额外加权（保持原有逻辑）
    yellow_red_mask = (sample_risk_labels >= 2) & (~positive_mask)
    sample_weights[yellow_red_mask] *= 5.0
    
    # 应用事件感知加权（如果提供了配置）
    if config is not None and hasattr(config.data, 'acceleration_events'):
        known_events = getattr(config.data, 'acceleration_events', [])
        if known_events:
            # 创建事件期间的掩码
            event_mask = np.zeros(len(sample_weights), dtype=bool)
            for event_start, event_end in known_events:
                start_dt = pd.to_datetime(event_start)
                end_dt = pd.to_datetime(event_end)
                event_period_mask = (sample_timestamps >= start_dt) & (sample_timestamps <= end_dt)
                event_mask |= event_period_mask
            
            # 对已知加速事件期间的样本应用额外权重
            additional_event_mask = event_mask & (~positive_mask) & (~yellow_red_mask)
            sample_weights[additional_event_mask] *= 3.0
    
    # 归一化权重
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    return sample_weights


def create_dataloaders(
    df_features: pd.DataFrame,
    df_quality: pd.DataFrame, 
    labels: pd.Series,
    config: 'PI_PHM_Config',
    event_aware_split: bool = True  # 新增参数
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        df_features: 特征DataFrame
        df_quality: 质量标记DataFrame  
        labels: 风险标签Series
        config: 配置对象
        event_aware_split: 是否使用事件感知划分
    """
    if event_aware_split:
        return _create_event_aware_dataloaders(df_features, df_quality, labels, config)
    else:
        return _create_standard_dataloaders(df_features, df_quality, labels, config)

def _create_standard_dataloaders(
    df_features: pd.DataFrame,
    df_quality: pd.DataFrame,
    labels: pd.Series,
    config: 'PI_PHM_Config'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """标准数据划分"""
    logger.info(f"_create_standard_dataloaders - df_features shape: {df_features.shape}, df_quality shape: {df_quality.shape}")
    
    # 验证输入数据
    assert len(df_features) == len(labels), "Features and labels must have the same length"
    assert len(df_features) == len(df_quality), "Features and quality must have the same length"
    assert list(df_features.index) == list(labels.index), "Features and labels must have the same index"
    assert list(df_features.index) == list(df_quality.index), "Features and quality must have the same index"
    
    # 构建静态特征
    static_features = create_static_features(config)
    
    # 时序划分
    train_end = pd.to_datetime(getattr(config.data, 'train_end', '2019-12-31'))
    val_start = pd.to_datetime(getattr(config.data, 'val_start', '2020-01-15'))
    val_end = pd.to_datetime(getattr(config.data, 'val_end', '2021-12-31'))
    test_start = pd.to_datetime(getattr(config.data, 'test_start', '2022-01-15'))
    
    # 检查数据时间范围
    data_start = df_features.index.min()
    data_end = df_features.index.max()
    logger.info(f"数据时间范围: {data_start} 到 {data_end}")
    logger.info(f"训练集范围: 到 {train_end}")
    logger.info(f"验证集范围: {val_start} 到 {val_end}")
    logger.info(f"测试集范围: 从 {test_start}")
    
    # 训练集掩码（排除gap）
    train_mask = df_features.index <= train_end
    
    # 验证集掩码
    val_mask = (df_features.index >= val_start) & (df_features.index <= val_end)
    
    # 测试集掩码
    test_mask = df_features.index >= test_start
    
    # 检查各数据集的时间范围是否有数据
    if not train_mask.any():
        logger.warning("训练集时间范围内没有数据!")
    if not val_mask.any():
        logger.warning("验证集时间范围内没有数据!")
    if not test_mask.any():
        logger.warning("测试集时间范围内没有数据!")
    
    # 构建事件检测标签（新增）
    from data.event_catalog import CreepBurstCatalog
    event_catalog = CreepBurstCatalog()
    y_event_all = generate_event_detection_labels(event_catalog, df_features, config.model.forecast)
    
    # 创建数据集实例，应用时间掩码
    train_dataset = AknesTimeSeriesDataset(
        df_features=df_features[train_mask],
        df_quality=df_quality[train_mask],
        labels=labels[train_mask],
        static_features=np.zeros(6),
        config=config,
        mode='train',
        y_event=y_event_all[train_mask]
    )
    
    val_dataset = AknesTimeSeriesDataset(
        df_features=df_features[val_mask],
        df_quality=df_quality[val_mask],
        labels=labels[val_mask],
        static_features=np.zeros(6),
        config=config,
        mode='val',
        y_event=y_event_all[val_mask]
    )
    
    test_dataset = AknesTimeSeriesDataset(
        df_features=df_features[test_mask],
        df_quality=df_quality[test_mask],
        labels=labels[test_mask],
        static_features=np.zeros(6),
        config=config,
        mode='test',
        y_event=y_event_all[test_mask]
    )
    
    logger.info(f"数据集构建完成!")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    if hasattr(train_dataset, 'sample_weights'):
        # 使用加权采样器
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def _create_event_aware_dataloaders(
    df_features: pd.DataFrame,
    df_quality: pd.DataFrame,
    labels: pd.Series,
    config: 'PI_PHM_Config'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    事件感知数据划分（方案A）- 任务5更新
    
    新划分：
    - 训练：2012-08-01 至 2021-06-30（包含约60个蠕变爆发事件，9.8年完整季节循环）
    - 验证：2021-07-15 至 2022-03-31（包含约10个蠕变爆发事件，覆盖秋冬季高发期）
    - 测试：2022-04-01 至 2022-12-25（包含约20个蠕变爆发事件，覆盖春夏秋半年周期）
    - Gap：各14天
    """
    logger.info("开始创建事件感知数据加载器...")
    
    # 构建静态特征
    static_features = create_static_features(config)
    
    # 定义新的时间边界（任务5要求）
    train_start = pd.to_datetime('2012-08-01')
    train_end = pd.to_datetime('2021-06-30')
    val_start = pd.to_datetime('2021-07-15') 
    val_end = pd.to_datetime('2022-03-31')
    test_start = pd.to_datetime('2022-04-01')
    test_end = pd.to_datetime('2022-12-25')
    
    logger.info(f"事件感知数据划分:")
    logger.info(f"  训练集: {train_start} 至 {train_end}")
    logger.info(f"  验证集: {val_start} 至 {val_end}")  
    logger.info(f"  测试集: {test_start} 至 {test_end}")
    
    # 创建掩码
    train_mask = (df_features.index >= train_start) & (df_features.index <= train_end)
    val_mask = (df_features.index >= val_start) & (df_features.index <= val_end)
    test_mask = (df_features.index >= test_start) & (df_features.index <= test_end)
    
    # 检查各数据集的时间范围是否有数据
    if not train_mask.any():
        logger.warning("训练集时间范围内没有数据!")
    if not val_mask.any():
        logger.warning("验证集时间范围内没有数据!")
    if not test_mask.any():
        logger.warning("测试集时间范围内没有数据!")
    
    # 构建事件检测标签（新增）
    from data.event_catalog import CreepBurstCatalog
    event_catalog = CreepBurstCatalog()
    y_event_all = generate_event_detection_labels(event_catalog, df_features, config.model.forecast)
    
    # 打印新划分下的统计信息（任务5要求）
    logger.info("新划分下各集的统计信息:")
    
    # 训练集统计
    train_events = event_catalog.get_events_in_date_range(train_start, train_end)
    train_y_event = y_event_all[train_mask]
    train_labels = labels[train_mask]
    logger.info(f"  训练集:")
    logger.info(f"    事件数: {len(train_events)}")
    if train_events:
        train_severity_counts = {'minor': 0, 'moderate': 0, 'major': 0}
        for event in train_events:
            train_severity_counts[event.severity] += 1
        logger.info(f"    事件severity分布: {train_severity_counts}")
    logger.info(f"    正样本（y_event=1）比例: {train_y_event.sum()}/{len(train_y_event)} ({train_y_event.mean():.2%})")
    logger.info(f"    标签分布: GREEN={(train_labels == 0).sum()}, BLUE={(train_labels == 1).sum()}, YELLOW={(train_labels == 2).sum()}, RED={(train_labels == 3).sum()}")
    
    # 验证集统计
    val_events = event_catalog.get_events_in_date_range(val_start, val_end)
    val_y_event = y_event_all[val_mask]
    val_labels = labels[val_mask]
    logger.info(f"  验证集:")
    logger.info(f"    事件数: {len(val_events)}")
    if val_events:
        val_severity_counts = {'minor': 0, 'moderate': 0, 'major': 0}
        for event in val_events:
            val_severity_counts[event.severity] += 1
        logger.info(f"    事件severity分布: {val_severity_counts}")
    logger.info(f"    正样本（y_event=1）比例: {val_y_event.sum()}/{len(val_y_event)} ({val_y_event.mean():.2%})")
    logger.info(f"    标签分布: GREEN={(val_labels == 0).sum()}, BLUE={(val_labels == 1).sum()}, YELLOW={(val_labels == 2).sum()}, RED={(val_labels == 3).sum()}")
    
    # 测试集统计
    test_events = event_catalog.get_events_in_date_range(test_start, test_end)
    test_y_event = y_event_all[test_mask]
    test_labels = labels[test_mask]
    logger.info(f"  测试集:")
    logger.info(f"    事件数: {len(test_events)}")
    if test_events:
        test_severity_counts = {'minor': 0, 'moderate': 0, 'major': 0}
        for event in test_events:
            test_severity_counts[event.severity] += 1
        logger.info(f"    事件severity分布: {test_severity_counts}")
    logger.info(f"    正样本（y_event=1）比例: {test_y_event.sum()}/{len(test_y_event)} ({test_y_event.mean():.2%})")
    logger.info(f"    标签分布: GREEN={(test_labels == 0).sum()}, BLUE={(test_labels == 1).sum()}, YELLOW={(test_labels == 2).sum()}, RED={(test_labels == 3).sum()}")
    
    # 创建数据集实例
    train_dataset = AknesTimeSeriesDataset(
        df_features=df_features[train_mask],
        df_quality=df_quality[train_mask] if df_quality is not None else None,
        labels=labels[train_mask],
        static_features=static_features,
        y_event=y_event_all[train_mask],  # 新增事件检测标签
        config=config,
        mode='train'
    )
    
    val_dataset = AknesTimeSeriesDataset(
        df_features=df_features[val_mask],
        df_quality=df_quality[val_mask] if df_quality is not None else None,
        labels=labels[val_mask],
        static_features=static_features,
        y_event=y_event_all[val_mask],  # 新增事件检测标签
        config=config,
        mode='val'
    )
    
    test_dataset = AknesTimeSeriesDataset(
        df_features=df_features[test_mask],
        df_quality=df_quality[test_mask] if df_quality is not None else None,
        labels=labels[test_mask],
        static_features=static_features,
        y_event=y_event_all[test_mask],  # 新增事件检测标签
        config=config,
        mode='test'
    )
    
    logger.info(f"数据集构建完成!")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")  
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    if hasattr(train_dataset, 'sample_weights'):
        # 使用加权采样器
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    dates = pd.date_range('2010-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # 模拟特征矩阵
    feature_cols = ['GNSS_12H', 'KH0206_Displacement', 'KH0206_Piezometer', 
                   'Seismicity_borehole', 'Air_temperature', 'velocity', 'acceleration', 
                   'anomaly', 'STA_LTA']
    df_features = pd.DataFrame(
        np.random.randn(1000, len(feature_cols)),
        index=dates,
        columns=feature_cols
    )
    
    # 模拟质量标记
    df_quality = pd.DataFrame(
        np.random.choice([0, 1, 2, 3], size=(1000, len(feature_cols)), p=[0.7, 0.15, 0.1, 0.05]),
        index=dates,
        columns=feature_cols
    )
    
    # 模拟风险标签
    labels = pd.Series(np.random.choice([0, 1, 2, 3], size=1000, p=[0.8, 0.12, 0.06, 0.02]), index=dates)
    
    # 创建模拟配置
    class MockConfig:
        class Data:
            train_end = '2019-12-31'
            val_start = '2020-01-15'
            val_end = '2021-12-31'
            test_start = '2022-01-15'
        class Model:
            avg_slope = 35.0
            joint_dip_direction = 125.0
            joint_dip_angle = 35.0
            cohesion = 50.0
            friction_angle = 31.0
            rock_density = 2700.0
            lookback = 60
            forecast = 7
        data = Data()
        model = Model()
    
    config = MockConfig()
    
    # 测试归一化器
    print("测试 PhysicsAwareNormalizer...")
    normalizer = PhysicsAwareNormalizer(config)
    train_mask = df_features.index <= '2019-12-31'
    normalizer.fit(df_features, train_mask)
    df_normalized = normalizer.transform(df_features)
    print(f"归一化前后形状: {df_features.shape} -> {df_normalized.shape}")
    
    # 测试数据集
    print("\n测试 AknesTimeSeriesDataset...")
    static_features = create_static_features(config)
    dataset = AknesTimeSeriesDataset(
        df_normalized, df_quality, labels, static_features,
        lookback=10, forecast=3, mode="train"
    )
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("样本键:", list(sample.keys()))
        print("x_dynamic shape:", sample["x_dynamic"].shape)
        print("x_static shape:", sample["x_static"].shape)
        print("y_risk:", sample["y_risk"])
    
    # 测试数据加载器
    print("\n测试 create_dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            df_features, df_quality, labels, config, batch_size=4
        )
        print(f"DataLoader 创建成功!")
        print(f"训练批次数: {len(train_loader)}")
        
        # 测试一个批次
        for batch in train_loader:
            print("批次键:", list(batch.keys()))
            print("x_dynamic batch shape:", batch["x_dynamic"].shape)
            break
            
    except Exception as e:
        print(f"DataLoader 测试失败: {e}")
    
    print("\n所有测试完成!")