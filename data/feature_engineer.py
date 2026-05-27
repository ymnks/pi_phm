"""特征工程模块 - 基于物理驱动的特征衍生"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AknesFeatureEngineer:
    """Åknes滑坡特征工程师"""
    
    def __init__(self, df_clean: pd.DataFrame, config: object):
        """
        初始化特征工程师
        
        Args:
            df_clean: 预处理后的数据DataFrame (N_days, 19)
            config: 配置对象
        """
        self.df_clean = df_clean.copy()
        self.config = config
        
        # 定义原始序列分组
        self.displacement_series = [
            'GNSS_12H', 'KH0206_Displacement', 'KH0112_Displacement', 
            'KH0117_Displacement', 'KH0118_Displacement', 'KH0217_Displacement', 'KH0218_Displacement', 'KH0306_Displacement'
        ]
        self.piezometer_series = [
            'KH0206_Piezometer', 'KH0112_Piezometer', 'KH0117_Piezometer',
            'KH0118_Piezometer', 'KH0217_Piezometer', 'KH0218_Piezometer', 'KH0306_Piezometer'
        ]
        self.seismic_series = ['Seismicity_borehole', 'Seismicity_surface']
        self.meteorological_series = ['Air_temperature', 'Precipitation', 'Snow', 'Surface_runoff']
        
        # 特征分组字典
        self.feature_groups = {
            'original': list(df_clean.columns),
            'group_A_deformation': [],
            'group_B_piezometer': [],
            'group_C_seismic': [],
            'group_D_meteorological': [],
            'group_E_cross_modal': []
        }
    
    def get_feature_index_map(self) -> Dict[str, List[int]]:
        """获取特征索引映射（任务6新增）"""
        feature_columns = list(self.df_clean.columns)
        index_map = {}
        
        # 位移序列索引
        displacement_indices = []
        for series in self.displacement_series:
            if series in feature_columns:
                displacement_indices.append(feature_columns.index(series))
        index_map['displacement'] = displacement_indices
        
        # 孔压序列索引
        piezometer_indices = []
        for series in self.piezometer_series:
            if series in feature_columns:
                piezometer_indices.append(feature_columns.index(series))
        index_map['piezometer'] = piezometer_indices
        
        # 微震序列索引
        seismic_indices = []
        for series in self.seismic_series:
            if series in feature_columns:
                seismic_indices.append(feature_columns.index(series))
        index_map['seismic'] = seismic_indices
        
        # 气象序列索引
        meteorological_indices = []
        for series in self.meteorological_series:
            if series in feature_columns:
                meteorological_indices.append(feature_columns.index(series))
        index_map['meteorological'] = meteorological_indices
        
        return index_map
    
    def transform(self) -> pd.DataFrame:
        """执行完整的特征工程流程"""
        logger.info("开始特征工程...")
        
        # 复制原始数据作为基础
        df_features = self.df_clean.copy()
        
        # 组A：形变衍生特征
        df_features = self._add_deformation_features(df_features)
        
        # 组B：孔压衍生特征  
        df_features = self._add_piezometer_features(df_features)
        
        # 组C：微震衍生特征
        df_features = self._add_seismic_features(df_features)
        
        # 组D：气象衍生特征
        df_features = self._add_meteorological_features(df_features)
        
        # 组E：跨模态耦合特征
        df_features = self._add_cross_modal_features(df_features)
        
        # 按组重新排列列顺序
        all_columns = (
            self.feature_groups['original'] +
            self.feature_groups['group_A_deformation'] +
            self.feature_groups['group_B_piezometer'] +
            self.feature_groups['group_C_seismic'] +
            self.feature_groups['group_D_meteorological'] +
            self.feature_groups['group_E_cross_modal']
        )
        
        # 只保留存在的列
        existing_columns = [col for col in all_columns if col in df_features.columns]
        df_features = df_features[existing_columns]
        
        # 打印统计信息
        total_channels = len(df_features.columns)
        logger.info(f"特征工程完成!")
        logger.info(f"各组特征数量:")
        logger.info(f"  原始特征: {len(self.feature_groups['original'])}")
        logger.info(f"  组A (形变): {len(self.feature_groups['group_A_deformation'])}")
        logger.info(f"  组B (孔压): {len(self.feature_groups['group_B_piezometer'])}")
        logger.info(f"  组C (微震): {len(self.feature_groups['group_C_seismic'])}")
        logger.info(f"  组D (气象): {len(self.feature_groups['group_D_meteorological'])}")
        logger.info(f"  组E (跨模态): {len(self.feature_groups['group_E_cross_modal'])}")
        logger.info(f"  总通道数 C_d: {total_channels}")
        
        return df_features
    
    def _add_deformation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加形变衍生特征（组A）"""
        df_out = df.copy()
        
        for series_name in self.displacement_series:
            if series_name not in df_out.columns:
                continue
                
            series = df_out[series_name].copy()
            
            # velocity：一阶差分 + 7天滑动均值平滑
            velocity_col = f"{series_name}_velocity"
            velocity = series.diff().rolling(window=7, min_periods=1).mean()
            df_out[velocity_col] = velocity
            self.feature_groups['group_A_deformation'].append(velocity_col)
            
            # acceleration：velocity的一阶差分 + 7天平滑
            acceleration_col = f"{series_name}_acceleration"
            acceleration = velocity.diff().rolling(window=7, min_periods=1).mean()
            df_out[acceleration_col] = acceleration
            self.feature_groups['group_A_deformation'].append(acceleration_col)
            
            # inverse_velocity：1/(|velocity| + ε)，ε=1e-8，并用99%分位数截断
            inv_velocity_col = f"{series_name}_inverse_velocity"
            epsilon = 1e-8
            inv_velocity = 1.0 / (np.abs(velocity) + epsilon)
            # 99%分位数截断
            quantile_99 = inv_velocity.quantile(0.99)
            inv_velocity = inv_velocity.clip(upper=quantile_99)
            df_out[inv_velocity_col] = inv_velocity
            self.feature_groups['group_A_deformation'].append(inv_velocity_col)
            
            # GNSS_12H额外特征
            if series_name == 'GNSS_12H':
                # tangent_angle：arctan(velocity / 1.0)
                tangent_col = f"{series_name}_tangent_angle"
                tangent_angle = np.arctan(velocity / 1.0)
                df_out[tangent_col] = tangent_angle
                self.feature_groups['group_A_deformation'].append(tangent_col)
                
                # jerk：acceleration的一阶差分
                jerk_col = f"{series_name}_jerk"
                jerk = acceleration.diff()
                df_out[jerk_col] = jerk
                self.feature_groups['group_A_deformation'].append(jerk_col)
                
                # creep_ratio：velocity / 30天滑动均值velocity
                creep_col = f"{series_name}_creep_ratio"
                velocity_30d_mean = velocity.rolling(window=30, min_periods=1).mean()
                creep_ratio = velocity / (velocity_30d_mean + epsilon)
                df_out[creep_col] = creep_ratio
                self.feature_groups['group_A_deformation'].append(creep_col)
                
                # STA_LTA：3天均值速率 / 30天均值速率
                sta_lta_col = f"{series_name}_STA_LTA"
                velocity_3d_mean = velocity.rolling(window=3, min_periods=1).mean()
                sta_lta = velocity_3d_mean / (velocity_30d_mean + epsilon)
                df_out[sta_lta_col] = sta_lta
                self.feature_groups['group_A_deformation'].append(sta_lta_col)
        
        return df_out
    
    def _add_piezometer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加孔压衍生特征（组B）"""
        df_out = df.copy()
        
        for series_name in self.piezometer_series:
            if series_name not in df_out.columns:
                continue
                
            series = df_out[series_name].copy()
            
            # piezometer_rate：一阶差分
            rate_col = f"{series_name}_rate"
            rate = series.diff()
            df_out[rate_col] = rate
            self.feature_groups['group_B_piezometer'].append(rate_col)
            
            # piezometer_7d_mean：7天滑动均值
            mean7d_col = f"{series_name}_7d_mean"
            mean7d = series.rolling(window=7, min_periods=1).mean()
            df_out[mean7d_col] = mean7d
            self.feature_groups['group_B_piezometer'].append(mean7d_col)
            
            # piezometer_anomaly：(当前值 - 30天均值) / (30天标准差 + 1e-6)
            anomaly_col = f"{series_name}_anomaly"
            mean30d = series.rolling(window=30, min_periods=1).mean()
            std30d = series.rolling(window=30, min_periods=1).std()
            anomaly = (series - mean30d) / (std30d + 1e-6)
            df_out[anomaly_col] = anomaly
            self.feature_groups['group_B_piezometer'].append(anomaly_col)
        
        # 整体指标
        valid_piezometer_cols = [col for col in self.piezometer_series if col in df_out.columns]
        if valid_piezometer_cols:
            # mean_piezometer：6个钻孔孔压的均值
            mean_piezometer = df_out[valid_piezometer_cols].mean(axis=1)
            df_out['mean_piezometer'] = mean_piezometer
            self.feature_groups['group_B_piezometer'].append('mean_piezometer')
            
            # max_piezometer：6个钻孔孔压的最大值
            max_piezometer = df_out[valid_piezometer_cols].max(axis=1)
            df_out['max_piezometer'] = max_piezometer
            self.feature_groups['group_B_piezometer'].append('max_piezometer')
            
            # mean_piezometer_rate：mean_piezometer的一阶差分
            mean_piezometer_rate = mean_piezometer.diff()
            df_out['mean_piezometer_rate'] = mean_piezometer_rate
            self.feature_groups['group_B_piezometer'].append('mean_piezometer_rate')
        
        return df_out
    
    def _add_seismic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加微震衍生特征（组C）"""
        df_out = df.copy()
        
        borehole_rate = None
        surface_rate = None
        
        for series_name in self.seismic_series:
            if series_name not in df_out.columns:
                continue
                
            series = df_out[series_name].copy()
            
            # seismic_rate：累计计数的一阶差分，clip(lower=0)
            rate_col = f"{series_name}_rate"
            rate = series.diff().clip(lower=0)
            df_out[rate_col] = rate
            self.feature_groups['group_C_seismic'].append(rate_col)
            
            # seismic_acceleration：seismic_rate的一阶差分
            accel_col = f"{series_name}_acceleration"
            accel = rate.diff()
            df_out[accel_col] = accel
            self.feature_groups['group_C_seismic'].append(accel_col)
            
            # seismic_7d_cumul：seismic_rate的7天滑动求和
            cumul7d_col = f"{series_name}_7d_cumul"
            cumul7d = rate.rolling(window=7, min_periods=1).sum()
            df_out[cumul7d_col] = cumul7d
            self.feature_groups['group_C_seismic'].append(cumul7d_col)
            
            # seismic_30d_cumul：seismic_rate的30天滑动求和
            cumul30d_col = f"{series_name}_30d_cumul"
            cumul30d = rate.rolling(window=30, min_periods=1).sum()
            df_out[cumul30d_col] = cumul30d
            self.feature_groups['group_C_seismic'].append(cumul30d_col)
            
            # seismic_STA_LTA：3天均值/30天均值
            sta_lta_col = f"{series_name}_STA_LTA"
            rate_3d_mean = rate.rolling(window=3, min_periods=1).mean()
            rate_30d_mean = rate.rolling(window=30, min_periods=1).mean()
            sta_lta = rate_3d_mean / (rate_30d_mean + 1e-8)
            df_out[sta_lta_col] = sta_lta
            self.feature_groups['group_C_seismic'].append(sta_lta_col)
            
            # seismic_energy_proxy：seismic_rate²的7天滑动求和（仅borehole）
            if series_name == 'Seismicity_borehole':
                energy_col = f"{series_name}_energy_proxy"
                energy_proxy = (rate ** 2).rolling(window=7, min_periods=1).sum()
                df_out[energy_col] = energy_proxy
                self.feature_groups['group_C_seismic'].append(energy_col)
                borehole_rate = rate
            
            if series_name == 'Seismicity_surface':
                surface_rate = rate
        
        # 额外生成特征
        if borehole_rate is not None:
            # seismic_total_rate：borehole_rate + surface_rate
            if surface_rate is not None:
                total_rate = borehole_rate.add(surface_rate, fill_value=0)
                df_out['seismic_total_rate'] = total_rate
                self.feature_groups['group_C_seismic'].append('seismic_total_rate')
            
            # seismic_depth_ratio：borehole_rate / (surface_rate + 1e-6)
            if surface_rate is not None:
                depth_ratio = borehole_rate / (surface_rate + 1e-6)
                df_out['seismic_depth_ratio'] = depth_ratio
                self.feature_groups['group_C_seismic'].append('seismic_depth_ratio')
        
        return df_out
    
    def _add_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加气象衍生特征（组D）"""
        df_out = df.copy()
        
        # 温度特征
        if 'Air_temperature' in df_out.columns:
            temp = df_out['Air_temperature'].copy()
            
            # temp_7d_mean：7天均温
            temp7d_col = 'temp_7d_mean'
            temp7d = temp.rolling(window=7, min_periods=1).mean()
            df_out[temp7d_col] = temp7d
            self.feature_groups['group_D_meteorological'].append(temp7d_col)
            
            # freeze_thaw_cycle：30天窗口内日均温穿越0°C的次数
            freeze_thaw_col = 'freeze_thaw_cycle'
            crossing_count = self._count_zero_crossings(temp, window=30)
            df_out[freeze_thaw_col] = crossing_count
            self.feature_groups['group_D_meteorological'].append(freeze_thaw_col)
            
            # positive_degree_day：30天窗口内正温度累加
            pos_deg_col = 'positive_degree_day'
            pos_temp = temp.clip(lower=0)
            pos_deg = pos_temp.rolling(window=30, min_periods=1).sum()
            df_out[pos_deg_col] = pos_deg
            self.feature_groups['group_D_meteorological'].append(pos_deg_col)
            
            # negative_degree_day：30天窗口内负温度绝对值累加
            neg_deg_col = 'negative_degree_day'
            neg_temp = (-temp).clip(lower=0)
            neg_deg = neg_temp.rolling(window=30, min_periods=1).sum()
            df_out[neg_deg_col] = neg_deg
            self.feature_groups['group_D_meteorological'].append(neg_deg_col)
        
        # 降水特征
        if 'Precipitation' in df_out.columns:
            precip = df_out['Precipitation'].copy().fillna(0)  # NaN当作0
            
            # rain_3d, rain_7d, rain_14d, rain_30d
            for days in [3, 7, 14, 30]:
                rain_col = f'rain_{days}d'
                rain_sum = precip.rolling(window=days, min_periods=1).sum()
                df_out[rain_col] = rain_sum
                self.feature_groups['group_D_meteorological'].append(rain_col)
            
            # antecedent_precip_index (API)：递推公式 API_t = 0.9 × API_{t-1} + P_t
            api_col = 'antecedent_precip_index'
            api = self._calculate_api(precip.values)
            df_out[api_col] = api
            self.feature_groups['group_D_meteorological'].append(api_col)
        
        # 积雪特征
        if 'Snow' in df_out.columns:
            snow = df_out['Snow'].copy()
            
            # snowmelt_rate：Snow差分的负值部分取绝对值
            melt_col = 'snowmelt_rate'
            snow_diff = snow.diff()
            melt_rate = (-snow_diff).clip(lower=0)
            df_out[melt_col] = melt_rate
            self.feature_groups['group_D_meteorological'].append(melt_col)
            
            # snow_accumulation：Snow差分的正值部分
            accum_col = 'snow_accumulation'
            accum_rate = snow_diff.clip(lower=0)
            df_out[accum_col] = accum_rate
            self.feature_groups['group_D_meteorological'].append(accum_col)
        
        # 径流特征
        if 'Surface_runoff' in df_out.columns:
            runoff = df_out['Surface_runoff'].copy()
            
            # runoff_7d_mean：7天均值
            runoff7d_col = 'runoff_7d_mean'
            runoff7d = runoff.rolling(window=7, min_periods=1).mean()
            df_out[runoff7d_col] = runoff7d
            self.feature_groups['group_D_meteorological'].append(runoff7d_col)
            
            # runoff_anomaly：(当前值 - 365天滑动均值) / (365天滑动标准差 + 1e-6)
            anomaly_col = 'runoff_anomaly'
            mean365d = runoff.rolling(window=365, min_periods=1).mean()
            std365d = runoff.rolling(window=365, min_periods=1).std()
            anomaly = (runoff - mean365d) / (std365d + 1e-6)
            df_out[anomaly_col] = anomaly
            self.feature_groups['group_D_meteorological'].append(anomaly_col)
        
        return df_out
    
    def _count_zero_crossings(self, series: pd.Series, window: int) -> pd.Series:
        """计算零穿越次数"""
        crossings = pd.Series(0, index=series.index, dtype=int)
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            # 计算符号变化次数
            signs = np.sign(window_data.values)
            # 处理零值：将其视为与前一个非零值相同符号
            for j in range(1, len(signs)):
                if signs[j] == 0:
                    signs[j] = signs[j-1] if signs[j-1] != 0 else 1
            
            crossing_count = np.sum(np.abs(np.diff(signs)) > 0)
            crossings.iloc[i] = crossing_count
        
        return crossings
    
    def _calculate_api(self, precip: np.ndarray) -> np.ndarray:
        """计算前期降水指数 (Antecedent Precipitation Index)"""
        api = np.zeros_like(precip)
        decay_factor = 0.9
        
        for i in range(1, len(precip)):
            api[i] = decay_factor * api[i-1] + precip[i]
        
        return api
    
    def _add_cross_modal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加跨模态耦合特征（组E）"""
        df_out = df.copy()
        epsilon = 1e-8
        
        # piezo_velocity_coupling：mean_piezometer_rate × |GNSS_velocity|
        if ('mean_piezometer_rate' in df_out.columns and 
            'GNSS_12H_velocity' in df_out.columns):
            coupling_col = 'piezo_velocity_coupling'
            coupling = df_out['mean_piezometer_rate'] * np.abs(df_out['GNSS_12H_velocity'])
            df_out[coupling_col] = coupling
            self.feature_groups['group_E_cross_modal'].append(coupling_col)
        
        # seismic_displacement_ratio：seismic_total_rate / (|GNSS_velocity| + 1e-8)
        if ('seismic_total_rate' in df_out.columns and 
            'GNSS_12H_velocity' in df_out.columns):
            ratio_col = 'seismic_displacement_ratio'
            ratio = df_out['seismic_total_rate'] / (np.abs(df_out['GNSS_12H_velocity']) + epsilon)
            df_out[ratio_col] = ratio
            self.feature_groups['group_E_cross_modal'].append(ratio_col)
        
        # effective_stress_proxy：1 - mean_piezometer / (expanding_max(mean_piezometer) + 1e-6)
        if 'mean_piezometer' in df_out.columns:
            stress_col = 'effective_stress_proxy'
            expanding_max = df_out['mean_piezometer'].expanding(min_periods=1).max()
            stress_proxy = 1 - df_out['mean_piezometer'] / (expanding_max + 1e-6)
            stress_proxy = stress_proxy.clip(lower=0, upper=1)
            df_out[stress_col] = stress_proxy
            self.feature_groups['group_E_cross_modal'].append(stress_col)
        
        # hydro_mech_lag7：rain_7d.shift(7) × |GNSS_velocity|
        if ('rain_7d' in df_out.columns and 
            'GNSS_12H_velocity' in df_out.columns):
            lag_col = 'hydro_mech_lag7'
            rain_lagged = df_out['rain_7d'].shift(7)
            lag_feature = rain_lagged * np.abs(df_out['GNSS_12H_velocity'])
            df_out[lag_col] = lag_feature
            self.feature_groups['group_E_cross_modal'].append(lag_col)
        
        return df_out
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """返回按组分类的特征名列表"""
        return self.feature_groups.copy()
    
    def get_total_channels(self) -> int:
        """返回总特征通道数 C_d"""
        return sum(len(features) for features in self.feature_groups.values())
    
    def plot_feature_correlation(self):
        """生成特征相关性热力图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 这里可以实现相关性热力图
            # 由于时间限制，先提供框架
            logger.info("plot_feature_correlation 功能待实现")
            
        except ImportError:
            logger.warning("matplotlib/seaborn 未安装，无法生成相关性热力图")


if __name__ == "__main__":
    # 测试代码
    from data_loader import AknesDataLoader
    from preprocessor import AknesPreprocessor
    
    class MockConfig:
        def __init__(self):
            self.data_dir = "/home/lab1111/zy/2024_AknesLandslide_Aspaas"
    
    # 1. 加载数据
    print("1. 加载原始数据...")
    loader = AknesDataLoader(MockConfig())
    raw_df = loader.load()
    print(f"原始数据形状: {raw_df.shape}")
    
    # 2. 预处理
    print("2. 数据预处理...")
    preprocessor = AknesPreprocessor(raw_df, MockConfig())
    clean_df, quality_flags = preprocessor.preprocess()
    print(f"预处理后数据形状: {clean_df.shape}")
    
    # 3. 特征工程
    print("3. 特征工程...")
    engineer = AknesFeatureEngineer(clean_df, MockConfig())
    feature_df = engineer.transform()
    print(f"特征工程后数据形状: {feature_df.shape}")
    
    # 4. 显示特征分组统计
    feature_groups = engineer.get_feature_groups()
    print("\n特征分组详情:")
    for group_name, features in feature_groups.items():
        print(f"{group_name}: {len(features)} 个特征")
    
    print(f"\n总特征通道数 C_d: {engineer.get_total_channels()}")