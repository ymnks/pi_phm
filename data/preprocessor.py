"""数据预处理模块 - 处理异常值、缺失值和物理一致性"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from scipy import interpolate

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AknesPreprocessor:
    """Åknes滑坡数据预处理器"""
    
    def __init__(self, df_raw: pd.DataFrame, config: Any):
        """
        初始化预处理器
        
        Args:
            df_raw: 原始数据DataFrame (N_days, 19)
            config: 配置对象
        """
        self.df_raw = df_raw.copy()
        self.config = config
        self.n_days, self.n_series = df_raw.shape
        
        # 定义序列类型分组
        self.displacement_series = [
            'GNSS_12H', 'KH0206_Displacement', 'KH0112_Displacement', 
            'KH0117_Displacement', 'KH0118_Displacement', 'KH0217_Displacement', 'KH0218_Displacement', 'KH0306_Displacement'
        ]
        self.piezometer_series = [
            'KH0206_Piezometer', 'KH0112_Piezometer', 'KH0117_Piezometer',
            'KH0118_Piezometer', 'KH0218_Piezometer', 'KH0306_Piezometer'
        ]
        self.seismic_series = ['Seismicity_borehole', 'Seismicity_surface']
        self.meteorological_series = ['Air_temperature', 'Precipitation', 'Snow', 'Surface_runoff']
        
        # 处理日志
        self.processing_log = {}
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行完整的预处理流程
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_clean, df_quality_flags)
        """
        logger.info("开始数据预处理...")
        
        # 初始化质量标记矩阵 (0 = 原始真实值)
        quality_flags = pd.DataFrame(
            np.zeros_like(self.df_raw.values, dtype=int),
            index=self.df_raw.index,
            columns=self.df_raw.columns
        )
        
        # 1. 异常值检测与处理
        df_processed = self._detect_and_handle_outliers(self.df_raw.copy())
        
        # 2. 缺失值插补
        df_interpolated, quality_flags = self._interpolate_missing_values(df_processed, quality_flags)
        
        # 3. 物理一致性校验
        self._validate_physical_consistency(df_interpolated)
        
        logger.info("数据预处理完成!")
        return df_interpolated, quality_flags
    
    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """异常值检测与处理"""
        df_out = df.copy()
        outlier_counts = {}
        
        # 处理位移累计序列
        for series_name in self.displacement_series:
            if series_name in df_out.columns:
                original_nans = df_out[series_name].isna().sum()
                series = df_out[series_name].copy()
                
                # 计算日变化量
                daily_diff = series.diff()
                
                # 检测单日突变超过 ±10mm 的点
                outlier_mask = (daily_diff.abs() > 10.0) & daily_diff.notna()
                outlier_indices = outlier_mask[outlier_mask].index
                
                # 标记为NaN
                if len(outlier_indices) > 0:
                    series.loc[outlier_indices] = np.nan
                    df_out[series_name] = series
                
                new_nans = series.isna().sum()
                outlier_counts[series_name] = new_nans - original_nans
                logger.info(f"位移序列 {series_name}: 检测到 {outlier_counts[series_name]} 个异常突变点")
        
        # 处理微震累计序列（计数器重置校正）
        for series_name in self.seismic_series:
            if series_name in df_out.columns:
                series = df_out[series_name].copy()
                if series.notna().any():
                    # 微震数据校正：检测负差分并校正
                    corrected_series = self._correct_seismic_counter_reset(series)
                    df_out[series_name] = corrected_series
                    
                    # 记录校正信息
                    outlier_counts[series_name] = "counter reset correction applied"
                    logger.info(f"微震序列 {series_name}: 已应用计数器重置校正")
        
        # 处理气象数据（3σ原则）
        for series_name in self.meteorological_series:
            if series_name in df_out.columns:
                series = df_out[series_name].copy()
                original_nans = series.isna().sum()
                
                # 计算均值和标准差（忽略NaN）
                mean_val = series.mean()
                std_val = series.std()
                
                # 3σ异常值检测
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                
                # 标记为NaN
                series.loc[outlier_mask] = np.nan
                df_out[series_name] = series
                
                new_nans = series.isna().sum()
                outlier_counts[series_name] = new_nans - original_nans
                logger.info(f"气象序列 {series_name}: 检测到 {outlier_counts[series_name]} 个3σ异常值")
        
        # 处理孔压数据（物理不合理值）
        # 注意：Water table [m bgl] 的物理含义是地面以下的水位深度，值天然为负数
        # 例如 -44.67m 表示水位在地面以下 44.67 米处
        # 因此不应过滤负值，只过滤极端异常值
        for series_name in self.piezometer_series:
            if series_name in df_out.columns:
                series = df_out[series_name].copy()
                original_nans = series.isna().sum()
                
                # 只过滤物理上不可能的值：
                # - 水位高于地面 (m bgl > 0) 极其罕见，视为异常
                # - 水位低于 -200m（超出钻孔深度范围）视为异常
                extreme_high_mask = (series > 0) & series.notna()
                series.loc[extreme_high_mask] = np.nan
                
                extreme_low_mask = (series < -200) & series.notna()
                series.loc[extreme_low_mask] = np.nan
                
                df_out[series_name] = series
                
                new_nans = series.isna().sum()
                outlier_counts[series_name] = new_nans - original_nans
                logger.info(f"孔压序列 {series_name}: 检测到 {outlier_counts[series_name]} 个极端异常值 (已保留负值水位数据)")
        
        self.processing_log['outliers'] = outlier_counts
        return df_out
    
    def _correct_seismic_counter_reset(self, series: pd.Series) -> pd.Series:
        """微震计数器重置校正算法"""
        if series.isna().all():
            return series
        
        # 移除NaN进行处理
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return series
        
        # 计算差分
        diff = valid_series.diff()
        
        # 检测负差分（阈值设为最小值的10%）
        threshold = valid_series.min() * 0.1 if valid_series.min() > 0 else 1.0
        reset_points = diff[diff < -threshold].index
        
        if len(reset_points) == 0:
            return series
        
        # 应用校正：在每个重置点后加上跳变量
        corrected_values = valid_series.copy()
        cumulative_offset = 0
        
        for i, reset_idx in enumerate(reset_points):
            # 找到重置前的值
            prev_idx = valid_series.index[valid_series.index.get_loc(reset_idx) - 1]
            prev_value = valid_series.loc[prev_idx]
            reset_value = valid_series.loc[reset_idx]
            
            # 计算跳变量（应该是正值）
            jump = prev_value - reset_value
            if jump > 0:
                cumulative_offset += jump
                
                # 找到重置点之后的所有索引
                after_reset_mask = valid_series.index > reset_idx
                if after_reset_mask.any():
                    corrected_values.loc[after_reset_mask] += cumulative_offset
        
        # 确保校正后的序列单调递增
        corrected_values = corrected_values.clip(lower=0)
        
        # 将校正值放回原序列
        result_series = series.copy()
        result_series.loc[corrected_values.index] = corrected_values
        
        return result_series
    
    def _interpolate_missing_values(self, df: pd.DataFrame, quality_flags: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """缺失值插补"""
        df_interp = df.copy()
        interp_quality_flags = quality_flags.copy()
        
        # 计算原始缺失率
        original_missing_rates = df_interp.isna().mean()
        
        for series_name in df_interp.columns:
            series = df_interp[series_name].copy()
            if series.isna().all():
                continue
            
            # 找到所有连续缺失段
            missing_segments = self._find_missing_segments(series)
            
            for start_idx, end_idx in missing_segments:
                segment_length = end_idx - start_idx + 1
                
                if segment_length <= 7:
                    # 短期缺失：三次样条插值
                    interpolated_values = self._cubic_spline_interpolate(series, start_idx, end_idx)
                    if interpolated_values is not None:
                        series.iloc[start_idx:end_idx+1] = interpolated_values
                        interp_quality_flags.iloc[start_idx:end_idx+1, 
                                                interp_quality_flags.columns.get_loc(series_name)] = 1
                
                elif segment_length <= 30:
                    # 中期缺失：尝试回归插补或线性插值
                    interpolated_values = self._regression_or_linear_interpolate(
                        df_interp, series_name, start_idx, end_idx
                    )
                    if interpolated_values is not None:
                        series.iloc[start_idx:end_idx+1] = interpolated_values
                        interp_quality_flags.iloc[start_idx:end_idx+1, 
                                                interp_quality_flags.columns.get_loc(series_name)] = 2
                
                # 长期缺失 (>30天)：保留NaN，quality_flag保持为0（但实际是缺失）
                # 在quality_flags中标记为3表示长期缺失
                else:
                    interp_quality_flags.iloc[start_idx:end_idx+1, 
                                            interp_quality_flags.columns.get_loc(series_name)] = 3
            
            df_interp[series_name] = series
        
        # 更新长期缺失的质量标记
        long_missing_mask = (df_interp.isna()) & (interp_quality_flags == 0)
        interp_quality_flags[long_missing_mask] = 3
        
        # 计算插补后缺失率
        final_missing_rates = df_interp.isna().mean()
        
        # 生成对比表格
        comparison_df = pd.DataFrame({
            'original_missing_rate': original_missing_rates,
            'final_missing_rate': final_missing_rates,
            'improvement': original_missing_rates - final_missing_rates
        })
        
        logger.info("缺失值插补完成，缺失率改善情况:")
        logger.info(comparison_df.to_string())
        
        self.processing_log['missing_value_improvement'] = comparison_df
        return df_interp, interp_quality_flags
    
    def _find_missing_segments(self, series: pd.Series) -> list:
        """找到所有连续缺失段"""
        if series.isna().all():
            return [(0, len(series)-1)]
        
        na_mask = series.isna()
        segments = []
        start = None
        
        for i, is_na in enumerate(na_mask):
            if is_na and start is None:
                start = i
            elif not is_na and start is not None:
                segments.append((start, i-1))
                start = None
        
        # 处理末尾的缺失段
        if start is not None:
            segments.append((start, len(series)-1))
        
        return segments
    
    def _cubic_spline_interpolate(self, series: pd.Series, start_idx: int, end_idx: int) -> np.ndarray:
        """三次样条插值"""
        # 需要足够的边界点
        window_size = min(10, start_idx, len(series) - 1 - end_idx)
        if window_size < 2:
            # 边界不足，使用线性插值
            return self._linear_interpolate(series, start_idx, end_idx)
        
        # 获取插值窗口
        left_start = max(0, start_idx - window_size)
        right_end = min(len(series), end_idx + window_size + 1)
        
        window_series = series.iloc[left_start:right_end]
        if window_series.notna().sum() < 4:  # 至少需要4个点进行样条插值
            return self._linear_interpolate(series, start_idx, end_idx)
        
        # 执行样条插值
        x_known = np.where(window_series.notna())[0]
        y_known = window_series.dropna().values
        x_interp = np.arange(start_idx - left_start, end_idx - left_start + 1)
        
        try:
            spline_func = interpolate.UnivariateSpline(x_known, y_known, k=min(3, len(x_known)-1))
            interpolated_values = spline_func(x_interp)
            return interpolated_values
        except Exception as e:
            logger.warning(f"样条插值失败: {e}, 使用线性插值")
            return self._linear_interpolate(series, start_idx, end_idx)
    
    def _linear_interpolate(self, series: pd.Series, start_idx: int, end_idx: int) -> np.ndarray:
        """线性插值"""
        return series.iloc[start_idx:end_idx+1].interpolate(method='linear').values
    
    def _regression_or_linear_interpolate(self, df: pd.DataFrame, series_name: str, start_idx: int, end_idx: int) -> np.ndarray:
        """回归插补或线性插值"""
        # 尝试找配对序列（同一钻孔的位移和孔压）
        paired_series = self._find_paired_series(series_name)
        
        if paired_series and paired_series in df.columns:
            # 使用配对序列进行简单回归
            paired_data = df[paired_series].iloc[start_idx:end_idx+1]
            if paired_data.notna().any():
                # 简单线性回归（使用历史数据）
                historical_mask = (df.index < df.index[start_idx]) & df[series_name].notna() & df[paired_series].notna()
                if historical_mask.sum() >= 10:  # 至少10个历史点
                    x_hist = df.loc[historical_mask, paired_series].values
                    y_hist = df.loc[historical_mask, series_name].values
                    
                    # 简单线性回归
                    slope, intercept = np.polyfit(x_hist, y_hist, 1)
                    interpolated_values = slope * paired_data.fillna(paired_data.mean()) + intercept
                    return interpolated_values.values
        
        # 回退到线性插值
        return self._linear_interpolate(df[series_name], start_idx, end_idx)
    
    def _find_paired_series(self, series_name: str) -> str:
        """找配对序列（同一钻孔的位移和孔压）"""
        if '_Displacement' in series_name:
            return series_name.replace('_Displacement', '_Piezometer')
        elif '_Piezometer' in series_name:
            return series_name.replace('_Piezometer', '_Displacement')
        return None
    
    def _validate_physical_consistency(self, df: pd.DataFrame):
        """物理一致性校验"""
        warnings = []
        
        # 检查位移序列的单调性
        for series_name in self.displacement_series:
            if series_name in df.columns:
                series = df[series_name].dropna()
                if len(series) > 3:
                    # 计算日速率
                    daily_rate = series.diff()
                    # 检查持续回退（>3天连续负速率）
                    negative_rates = daily_rate < 0
                    consecutive_negative = self._find_consecutive_trues(negative_rates, min_length=3)
                    if consecutive_negative:
                        warnings.append(f"位移序列 {series_name} 存在持续回退现象")
        
        # 检查微震序列的单调性
        for series_name in self.seismic_series:
            if series_name in df.columns:
                series = df[series_name].dropna()
                if len(series) > 1:
                    diff = series.diff().iloc[1:]  # 跳过第一个NaN
                    if (diff < 0).any():
                        warnings.append(f"微震序列 {series_name} 未严格单调递增")
        
        # 检查孔压范围（Water table [m bgl] 天然为负值，合理范围约 [-200, 0]）
        for series_name in self.piezometer_series:
            if series_name in df.columns:
                series = df[series_name].dropna()
                if len(series) > 0:
                    if (series > 0).any() or (series < -200).any():
                        warnings.append(f"孔压序列 {series_name} 超出合理范围 [-200, 0] m bgl")
        
        # 输出警告
        for warning in warnings:
            logger.warning(warning)
        
        self.processing_log['physical_consistency_warnings'] = warnings
    
    def _find_consecutive_trues(self, boolean_series: pd.Series, min_length: int) -> bool:
        """检查是否存在连续True序列"""
        if len(boolean_series) == 0:
            return False
        
        current_streak = 0
        for val in boolean_series:
            if val:
                current_streak += 1
                if current_streak >= min_length:
                    return True
            else:
                current_streak = 0
        return False
    
    def plot_before_after(self):
        """生成预处理前后对比图"""
        try:
            import matplotlib.pyplot as plt
            
            # 这里可以实现可视化对比
            # 由于时间限制，先提供框架
            logger.info("plot_before_after 功能待实现")
            
        except ImportError:
            logger.warning("matplotlib 未安装，无法生成对比图")


if __name__ == "__main__":
    # 简单测试
    from data_loader import AknesDataLoader
    
    class MockConfig:
        def __init__(self):
            self.data_dir = "/home/lab1111/zy/2024_AknesLandslide_Aspaas"
    
    # 加载原始数据
    loader = AknesDataLoader(MockConfig())
    raw_df = loader.load()
    
    # 预处理
    preprocessor = AknesPreprocessor(raw_df, MockConfig())
    clean_df, quality_flags = preprocessor.preprocess()
    
    print(f"原始数据形状: {raw_df.shape}")
    print(f"预处理后数据形状: {clean_df.shape}")
    print(f"质量标记形状: {quality_flags.shape}")
    print(f"预处理后总缺失率: {clean_df.isna().mean().mean():.2%}")