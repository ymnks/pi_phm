"""风险标签生成模块 - 基于多源数据自动生成四级风险等级"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLabelGenerator:
    """风险标签生成器"""
    
    def __init__(self, df_features: pd.DataFrame, config: object):
        """
        初始化风险标签生成器
        
        Args:
            df_features: 特征矩阵DataFrame
            config: 配置对象，包含已知加速事件等信息
        """
        self.df_features = df_features.copy()
        self.config = config
        
        # 初始阈值（基于Prompt 4的定义）
        self.thresholds = {
            'displacement': {'green': 0.15, 'blue': 0.30, 'yellow': 0.60},
            'seismic': {'green': 5, 'blue': 15, 'yellow': 40},
            'piezometer': {'green': 0.3, 'blue': 0.8, 'yellow': 1.5}
        }
        
        # 风险等级映射
        self.risk_levels = {0: 'GREEN', 1: 'BLUE', 2: 'YELLOW', 3: 'RED'}
    
    def _compute_conservative_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        使用保守的固定阈值（基于Prompt 4的原始定义）
        """
        logger.info("使用保守的固定阈值...")
        return {
            'displacement': {'green': 0.15, 'blue': 0.30, 'yellow': 0.60},
            'seismic': {'green': 5, 'blue': 15, 'yellow': 40},
            'piezometer': {'green': 0.3, 'blue': 0.8, 'yellow': 1.5}
        }
    
    def _compute_dynamic_thresholds(self, train_mask: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        基于训练集数据分布动态计算风险阈值
        
        Args:
            train_mask: 训练集掩码
            
        Returns:
            动态阈值字典
        """
        logger.info("基于训练集数据分布动态计算风险阈值...")
        
        dynamic_thresholds = {}
        
        # 位移速率阈值 - 使用分位数
        if 'GNSS_12H_velocity' in self.df_features.columns:
            velocity_train = self.df_features.loc[train_mask, 'GNSS_12H_velocity'].abs().dropna()
            if len(velocity_train) > 0:
                # 使用分位数：GREEN < 75%, BLUE < 90%, YELLOW < 95%
                green_thresh = velocity_train.quantile(0.75)
                blue_thresh = velocity_train.quantile(0.90) 
                yellow_thresh = velocity_train.quantile(0.95)
                dynamic_thresholds['displacement'] = {
                    'green': max(green_thresh, 0.01),  # 确保最小阈值
                    'blue': max(blue_thresh, 0.05),
                    'yellow': max(yellow_thresh, 0.1)
                }
                logger.info(f"位移速率动态阈值: GREEN<{green_thresh:.4f}, BLUE<{blue_thresh:.4f}, YELLOW<{yellow_thresh:.4f}")
            else:
                dynamic_thresholds['displacement'] = self.thresholds['displacement']
                logger.warning("训练集中无有效位移速率数据，使用默认阈值")
        else:
            dynamic_thresholds['displacement'] = self.thresholds['displacement']
            logger.warning("未找到GNSS_12H_velocity列，使用默认阈值")
        
        # 微震事件率阈值
        if 'seismic_total_rate' in self.df_features.columns:
            seismic_train = self.df_features.loc[train_mask, 'seismic_total_rate'].dropna()
            if len(seismic_train) > 0:
                green_thresh = seismic_train.quantile(0.75)
                blue_thresh = seismic_train.quantile(0.90)
                yellow_thresh = seismic_train.quantile(0.95)
                dynamic_thresholds['seismic'] = {
                    'green': max(green_thresh, 1.0),
                    'blue': max(blue_thresh, 5.0),
                    'yellow': max(yellow_thresh, 10.0)
                }
                logger.info(f"微震事件率动态阈值: GREEN<{green_thresh:.2f}, BLUE<{blue_thresh:.2f}, YELLOW<{yellow_thresh:.2f}")
            else:
                dynamic_thresholds['seismic'] = self.thresholds['seismic']
                logger.warning("训练集中无有效微震数据，使用默认阈值")
        else:
            dynamic_thresholds['seismic'] = self.thresholds['seismic']
            logger.warning("未找到seismic_total_rate列，使用默认阈值")
        
        # 孔压变化率阈值
        if 'mean_piezometer_rate' in self.df_features.columns:
            piezo_train = self.df_features.loc[train_mask, 'mean_piezometer_rate'].abs().dropna()
            if len(piezo_train) > 0:
                green_thresh = piezo_train.quantile(0.75)
                blue_thresh = piezo_train.quantile(0.90)
                yellow_thresh = piezo_train.quantile(0.95)
                dynamic_thresholds['piezometer'] = {
                    'green': max(green_thresh, 0.1),
                    'blue': max(blue_thresh, 0.3),
                    'yellow': max(yellow_thresh, 0.5)
                }
                logger.info(f"孔压变化率动态阈值: GREEN<{green_thresh:.4f}, BLUE<{blue_thresh:.4f}, YELLOW<{yellow_thresh:.4f}")
            else:
                dynamic_thresholds['piezometer'] = self.thresholds['piezometer']
                logger.warning("训练集中无有效孔压数据，使用默认阈值")
        else:
            dynamic_thresholds['piezometer'] = self.thresholds['piezometer']
            logger.warning("未找到mean_piezometer_rate列，使用默认阈值")
        
        return dynamic_thresholds
    
    def generate(self, method: str = "max_fusion", train_mask: pd.Series = None, use_conservative: bool = False) -> pd.Series:
        """
        生成风险等级标签
        
        Args:
            method: 融合方法 ("max_fusion", "weighted_vote", "cascade")
            train_mask: 训练集掩码，用于动态阈值计算
            use_conservative: 是否使用保守阈值
            
        Returns:
            pd.Series: 风险等级标签 (0-3)
        """
        logger.info(f"使用 {method} 方法生成风险标签...")
        
        # 根据配置选择阈值策略
        if use_conservative:
            self.thresholds = self._compute_conservative_thresholds()
        elif train_mask is not None:
            self.thresholds = self._compute_dynamic_thresholds(train_mask)
        
        # 1. 单源判级
        displacement_labels = self._classify_by_displacement()
        seismic_labels = self._classify_by_seismic()
        piezometer_labels = self._classify_by_piezometer()
        
        # 2. 多源融合
        if method == "max_fusion":
            labels = self._max_fusion(displacement_labels, seismic_labels, piezometer_labels)
        elif method == "weighted_vote":
            labels = self._weighted_vote(displacement_labels, seismic_labels, piezometer_labels)
        elif method == "cascade":
            labels = self._cascade_fusion(displacement_labels, seismic_labels, piezometer_labels)
        else:
            raise ValueError(f"未知的融合方法: {method}")
        
        # 3. 时间平滑
        labels = self._time_smoothing(labels)
        
        # 4. 确保标签在有效范围内
        labels = labels.clip(lower=0, upper=3).astype(int)
        
        logger.info("风险标签生成完成!")
        return labels
    
    def _classify_by_displacement(self) -> pd.Series:
        """基于位移速率的单源判级"""
        if 'GNSS_12H_velocity' not in self.df_features.columns:
            logger.warning("GNSS_12H_velocity 列不存在，返回全GREEN标签")
            return pd.Series(0, index=self.df_features.index)
        
        velocity = self.df_features['GNSS_12H_velocity'].abs()
        thresholds = self.thresholds['displacement']
        
        labels = pd.Series(0, index=self.df_features.index, dtype=int)
        labels[velocity > thresholds['yellow']] = 3  # RED
        labels[(velocity <= thresholds['yellow']) & (velocity > thresholds['blue'])] = 2  # YELLOW
        labels[(velocity <= thresholds['blue']) & (velocity > thresholds['green'])] = 1  # BLUE
        # velocity <= green 保持为 0 (GREEN)
        
        return labels
    
    def _classify_by_seismic(self) -> pd.Series:
        """基于微震事件率的单源判级"""
        if 'seismic_total_rate' not in self.df_features.columns:
            logger.warning("seismic_total_rate 列不存在，返回全GREEN标签")
            return pd.Series(0, index=self.df_features.index)
        
        rate = self.df_features['seismic_total_rate']
        thresholds = self.thresholds['seismic']
        
        labels = pd.Series(0, index=self.df_features.index, dtype=int)
        labels[rate > thresholds['yellow']] = 3  # RED
        labels[(rate <= thresholds['yellow']) & (rate > thresholds['blue'])] = 2  # YELLOW
        labels[(rate <= thresholds['blue']) & (rate > thresholds['green'])] = 1  # BLUE
        
        return labels
    
    def _classify_by_piezometer(self) -> pd.Series:
        """基于孔压变化率的单源判级"""
        if 'mean_piezometer_rate' not in self.df_features.columns:
            logger.warning("mean_piezometer_rate 列不存在，返回全GREEN标签")
            return pd.Series(0, index=self.df_features.index)
        
        rate = self.df_features['mean_piezometer_rate'].abs()
        thresholds = self.thresholds['piezometer']
        
        labels = pd.Series(0, index=self.df_features.index, dtype=int)
        labels[rate > thresholds['yellow']] = 3  # RED
        labels[(rate <= thresholds['yellow']) & (rate > thresholds['blue'])] = 2  # YELLOW
        labels[(rate <= thresholds['blue']) & (rate > thresholds['green'])] = 1  # BLUE
        
        return labels
    
    def _max_fusion(self, disp_labels: pd.Series, seismic_labels: pd.Series, 
                   piezo_labels: pd.Series) -> pd.Series:
        """最大值融合（最保守）"""
        # 处理NaN情况：如果某个源全是NaN，则不参与融合
        sources = []
        if not disp_labels.isna().all():
            sources.append(disp_labels)
        if not seismic_labels.isna().all():
            sources.append(seismic_labels)
        if not piezo_labels.isna().all():
            sources.append(piezo_labels)
        
        if len(sources) == 0:
            return pd.Series(0, index=self.df_features.index)
        elif len(sources) == 1:
            return sources[0].fillna(0).astype(int)
        else:
            # 取最大值
            combined = pd.concat(sources, axis=1)
            return combined.max(axis=1).fillna(0).astype(int)
    
    def _weighted_vote(self, disp_labels: pd.Series, seismic_labels: pd.Series, 
                      piezo_labels: pd.Series) -> pd.Series:
        """加权投票融合"""
        weights = {'displacement': 0.5, 'seismic': 0.3, 'piezometer': 0.2}
        
        # 构建权重矩阵
        weighted_sum = pd.Series(0.0, index=self.df_features.index)
        total_weight = pd.Series(0.0, index=self.df_features.index)
        
        sources_and_weights = [
            (disp_labels, weights['displacement']),
            (seismic_labels, weights['seismic']),
            (piezo_labels, weights['piezometer'])
        ]
        
        for source, weight in sources_and_weights:
            if not source.isna().all():
                weighted_sum += source * weight
                total_weight += weight
        
        # 处理没有有效源的情况
        valid_mask = total_weight > 0
        final_labels = pd.Series(0, index=self.df_features.index, dtype=int)
        final_labels[valid_mask] = (weighted_sum[valid_mask] / total_weight[valid_mask]).round().astype(int)
        
        return final_labels.clip(lower=0, upper=3).astype(int)
    
    def _cascade_fusion(self, disp_labels: pd.Series, seismic_labels: pd.Series, 
                       piezo_labels: pd.Series) -> pd.Series:
        """级联融合策略"""
        labels = disp_labels.copy().fillna(0).astype(int)
        
        # 如果微震也支持（微震等级≥位移等级且位移等级≥1），等级+1（不超过RED）
        seismic_valid = ~seismic_labels.isna()
        condition1 = seismic_valid & (seismic_labels >= labels) & (labels >= 1)
        labels[condition1] = (labels[condition1] + 1).clip(upper=3)
        
        # 如果孔压异常（孔压等级≥2）但位移低（等级=0），提升到BLUE
        piezo_valid = ~piezo_labels.isna()
        condition2 = piezo_valid & (piezo_labels >= 2) & (labels == 0)
        labels[condition2] = 1
        
        return labels.astype(int)
    
    def _time_smoothing(self, labels: pd.Series) -> pd.Series:
        """时间平滑处理"""
        # 3天窗口中位数滤波
        smoothed_labels = labels.rolling(window=3, center=True, min_periods=1).median()
        
        # 处理从GREEN直接跳到YELLOW以上的情况，插入BLUE过渡
        smoothed_array = smoothed_labels.values.copy()
        for i in range(1, len(smoothed_array)):
            if smoothed_array[i-1] == 0 and smoothed_array[i] >= 2:
                # 插入BLUE过渡（如果可能）
                if i > 0:
                    smoothed_array[i-1] = 1
        
        # 确保数据类型为整数
        smoothed_array = np.round(smoothed_array).astype(int)
        return pd.Series(smoothed_array, index=labels.index, dtype=int)
    
    def calibrate_thresholds(self, train_end_date: Optional[str] = None) -> Dict:
        """
        自动校准阈值
        
        Args:
            train_end_date: 可选的训练结束日期，如果提供则使用该日期，否则使用config中的日期
            
        Returns:
            Dict: 校准后的阈值字典
        """
        logger.info("开始自动校准阈值...")
        
        if train_end_date is not None:
            train_end = pd.to_datetime(train_end_date)
            logger.info(f"使用自定义训练结束日期: {train_end}")
        else:
            # 只使用训练集时间范围的数据（2010-01-01至2019-12-31）
            train_end = pd.to_datetime(getattr(self.config.data, 'train_end', '2019-12-31'))
        
        train_mask = self.df_features.index <= train_end
        train_data = self.df_features[train_mask]
        
        calibrated_thresholds = {}
        
        # 位移速率校准
        if 'GNSS_12H_velocity' in train_data.columns:
            velocity_abs = train_data['GNSS_12H_velocity'].abs().dropna()
            if len(velocity_abs) > 0:
                p75, p90, p99 = velocity_abs.quantile([0.75, 0.90, 0.99])
                calibrated_thresholds['displacement'] = {
                    'green': p75,
                    'blue': p90,
                    'yellow': p99
                }
            else:
                calibrated_thresholds['displacement'] = self.thresholds['displacement']
        else:
            calibrated_thresholds['displacement'] = self.thresholds['displacement']
        
        # 微震事件率校准
        if 'seismic_total_rate' in train_data.columns:
            seismic_rate = train_data['seismic_total_rate'].dropna()
            if len(seismic_rate) > 0:
                p75, p90, p99 = seismic_rate.quantile([0.75, 0.90, 0.99])
                calibrated_thresholds['seismic'] = {
                    'green': p75,
                    'blue': p90,
                    'yellow': p99
                }
            else:
                calibrated_thresholds['seismic'] = self.thresholds['seismic']
        else:
            calibrated_thresholds['seismic'] = self.thresholds['seismic']
        
        # 孔压变化率校准
        if 'mean_piezometer_rate' in train_data.columns:
            piezo_rate_abs = train_data['mean_piezometer_rate'].abs().dropna()
            if len(piezo_rate_abs) > 0:
                p75, p90, p99 = piezo_rate_abs.quantile([0.75, 0.90, 0.99])
                calibrated_thresholds['piezometer'] = {
                    'green': p75,
                    'blue': p90,
                    'yellow': p99
                }
            else:
                calibrated_thresholds['piezometer'] = self.thresholds['piezometer']
        else:
            calibrated_thresholds['piezometer'] = self.thresholds['piezometer']
        
        # 更新内部阈值
        self.thresholds = calibrated_thresholds
        
        logger.info("阈值校准完成!")
        return calibrated_thresholds
    
    def validate_against_known_events(self, labels: pd.Series) -> Dict:
        """
        与已知加速事件进行交叉验证
        
        Args:
            labels: 生成的风险标签
            
        Returns:
            Dict: 验证报告
        """
        logger.info("开始与已知加速事件交叉验证...")
        
        validation_report = {}
        known_events = getattr(self.config.data, 'acceleration_events', [])
        
        if not known_events:
            logger.warning("配置中未找到已知加速事件")
            return {"warning": "无已知事件用于验证"}
        
        yellow_or_higher_count = 0
        total_event_days = 0
        
        for i, (start_date, end_date) in enumerate(known_events):
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 获取事件期间的标签
            event_mask = (labels.index >= start_dt) & (labels.index <= end_dt)
            event_labels = labels[event_mask]
            
            if len(event_labels) == 0:
                continue
                
            total_event_days += len(event_labels)
            yellow_or_higher = (event_labels >= 2).sum()
            yellow_or_higher_count += yellow_or_higher
            
            # 计算事件期间YELLOW及以上标签的比例
            yellow_ratio = yellow_or_higher / len(event_labels)
            validation_report[f"event_{i+1}"] = {
                'period': f"{start_date} to {end_date}",
                'total_days': len(event_labels),
                'yellow_or_higher_days': int(yellow_or_higher),
                'yellow_ratio': float(yellow_ratio),
                'meets_requirement': yellow_ratio >= 0.5
            }
            
            if yellow_ratio < 0.5:
                logger.warning(f"事件 {i+1} 不满足要求: YELLOW比例={yellow_ratio:.2%} < 50%")
        
        overall_ratio = yellow_or_higher_count / total_event_days if total_event_days > 0 else 0
        validation_report['overall'] = {
            'total_event_days': total_event_days,
            'total_yellow_or_higher': yellow_or_higher_count,
            'overall_yellow_ratio': float(overall_ratio),
            'recommendation': "建议调低阈值" if overall_ratio < 0.5 else "阈值设置合理"
        }
        
        logger.info("交叉验证完成!")
        return validation_report
    
    def get_label_statistics(self, labels: pd.Series) -> Dict:
        """
        获取标签分布统计
        
        Args:
            labels: 风险标签
            
        Returns:
            Dict: 标签统计信息
        """
        value_counts = labels.value_counts().sort_index()
        total = len(labels)
        
        statistics = {
            'total_samples': total,
            'distribution': {},
            'percentages': {}
        }
        
        for level in range(4):
            count = value_counts.get(level, 0)
            percentage = count / total * 100
            statistics['distribution'][self.risk_levels[level]] = int(count)
            statistics['percentages'][self.risk_levels[level]] = float(percentage)
        
        return statistics
    
    def plot_labels(self, labels: pd.Series):
        """生成可视化（框架实现）"""
        try:
            import matplotlib.pyplot as plt
            
            # 这里可以实现完整的可视化
            # 由于时间限制，先提供框架
            logger.info("plot_labels 功能待实现")
            
        except ImportError:
            logger.warning("matplotlib 未安装，无法生成可视化")
    
    def generate_event_based_labels(self, df_features: pd.DataFrame) -> pd.Series:
        """
        基于真实蠕变爆发事件生成四级风险标签
        
        标签策略（基于Aspaas et al., 2024论文）：
        GREEN (0) - 正常蠕变：
          没有任何蠕变爆发事件的日期
          且不在任何事件的前驱窗口内

        BLUE (1) - 关注期：
          处于某个事件的前驱窗口内（事件开始前7-14天）
          或处于minor事件期间

        YELLOW (2) - 警戒期：
          处于moderate事件期间
          或处于major事件的前驱窗口内（事件开始前3-7天）
          或处于多钻孔同步事件期间（>=3个钻孔同时活跃）

        RED (3) - 危险期：
          处于major事件期间
          或处于多钻孔同步的moderate+事件期间

        Args:
            df_features: 特征DataFrame，索引为日期
            
        Returns:
            与df_features同index的标签序列
        """
        from data.event_catalog import CreepBurstCatalog
        
        # 加载事件目录
        catalog = CreepBurstCatalog()
        if not catalog.events:
            logger.warning("No events loaded, falling back to original label generation")
            return self.generate(method="max_fusion")
        
        # 初始化标签为GREEN
        labels = pd.Series(0, index=df_features.index, dtype=int)
        
        # 统计信息
        total_days = len(df_features)
        event_covered = set()
        sync_events_count = 0
        
        # 遍历每个日期
        for date in df_features.index:
            if not isinstance(date, pd.Timestamp):
                continue
                
            # 检查是否在任何事件期间
            current_events = []
            for event in catalog.events:
                if event.start_time.date() <= date.date() <= event.end_time.date():
                    current_events.append(event)
                    event_covered.add(date)
            
            # 检查同步性
            sync_count = catalog.count_sync_boreholes(date, window_days=3)
            is_high_sync = sync_count >= 3
            is_very_high_sync = sync_count >= 5
            
            # 检查前驱窗口
            precursor_events_7_14 = []  # 7-14天前驱
            precursor_events_3_7 = []   # 3-7天前驱
            for event in catalog.events:
                days_before = (event.start_time.date() - date.date()).days
                if 7 <= days_before <= 14:
                    precursor_events_7_14.append(event)
                elif 3 <= days_before <= 7:
                    precursor_events_3_7.append(event)
            
            # 确定标签
            current_label = 0  # 默认GREEN
            
            # 检查RED条件
            red_conditions = []
            for event in current_events:
                if event.severity == "major":
                    red_conditions.append(True)
                elif event.severity in ["moderate", "major"] and is_high_sync:
                    red_conditions.append(True)
            
            if red_conditions:
                current_label = 3
            else:
                # 检查YELLOW条件
                yellow_conditions = []
                for event in current_events:
                    if event.severity == "moderate":
                        yellow_conditions.append(True)
                
                # major事件的3-7天前驱
                for event in precursor_events_3_7:
                    if event.severity == "major":
                        yellow_conditions.append(True)
                
                # 多钻孔同步事件
                if is_high_sync and current_events:
                    yellow_conditions.append(True)
                
                if yellow_conditions:
                    current_label = 2
                else:
                    # 检查BLUE条件
                    blue_conditions = []
                    # minor事件期间
                    for event in current_events:
                        if event.severity == "minor":
                            blue_conditions.append(True)
                    
                    # 7-14天前驱窗口
                    if precursor_events_7_14:
                        blue_conditions.append(True)
                    
                    if blue_conditions:
                        current_label = 1
            
            labels[date] = current_label
        
        # 输出统计信息
        label_counts = labels.value_counts().sort_index()
        green_count = label_counts.get(0, 0)
        blue_count = label_counts.get(1, 0)
        yellow_count = label_counts.get(2, 0)
        red_count = label_counts.get(3, 0)
        
        print(f"基于事件的标签生成完成:")
        print(f"  总天数: {total_days}")
        print(f"  GREEN: {green_count}天 ({green_count/total_days*100:.1f}%)")
        print(f"  BLUE:  {blue_count}天 ({blue_count/total_days*100:.1f}%)")
        print(f"  YELLOW: {yellow_count}天 ({yellow_count/total_days*100:.1f}%)")
        print(f"  RED:   {red_count}天 ({red_count/total_days*100:.1f}%)")
        print(f"  覆盖的蠕变爆发事件数: {len(event_covered)} / {len(catalog.events)}")
        
        # 统计同步事件
        sync_days = 0
        for date in df_features.index:
            if not isinstance(date, pd.Timestamp):
                continue
            sync_count = catalog.count_sync_boreholes(date, window_days=3)
            if sync_count >= 3:
                sync_days += 1
        
        print(f"  同步事件天数: {sync_days}")
        
        return labels

    # 添加generate_labels方法，以兼容现有代码
    def generate_labels(self, df_features: pd.DataFrame = None, acceleration_events: List[tuple] = None):
        """
        生成风险标签的兼容方法，用于向后兼容
        
        Args:
            df_features: 特征DataFrame (如果未在初始化时提供)
            acceleration_events: 加速事件列表 (如果未在config中提供)
        
        Returns:
            pd.Series: 风险等级标签 (0-3)
        """
        logger.info("使用兼容的generate_labels方法生成风险标签...")
        
        # 如果提供了新的特征数据，更新内部数据
        if df_features is not None:
            self.df_features = df_features.copy()
        
        # 使用默认的最大融合方法生成标签
        labels = self.generate(method="max_fusion")
        
        logger.info("兼容方法生成风险标签完成!")
        return labels


if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    import numpy as np
    
    # 创建模拟特征数据
    dates = pd.date_range('2010-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    test_features = pd.DataFrame({
        'GNSS_12H_velocity': np.random.normal(0.2, 0.1, 1000),
        'seismic_total_rate': np.random.poisson(10, 1000),
        'mean_piezometer_rate': np.random.normal(0.5, 0.2, 1000)
    }, index=dates)
    
    # 创建模拟配置
    class MockConfig:
        class Data:
            train_end = '2019-12-31'
            acceleration_events = [
                ('2011-05-01', '2011-07-15'),
                ('2013-10-15', '2013-12-01'),
                ('2015-04-10', '2015-06-20')
            ]
        data = Data()
    
    # 测试风险标签生成
    generator = RiskLabelGenerator(test_features, MockConfig())
    
    # 测试阈值校准
    calibrated_thresholds = generator.calibrate_thresholds()
    print("校准后的阈值:", calibrated_thresholds)
    
    # 测试标签生成
    labels = generator.generate(method="max_fusion")
    print(f"生成的标签形状: {labels.shape}")
    print(f"标签分布: {labels.value_counts().sort_index()}")
    
    # 测试统计
    stats = generator.get_label_statistics(labels)
    print("标签统计:", stats)
    
    # 测试验证
    validation = generator.validate_against_known_events(labels)
    print("验证结果:", validation)
    
    print("风险标签生成器测试完成!")