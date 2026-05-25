"""数据加载模块 - 自适应处理多种文件格式和数据结构"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AknesDataLoader:
    """Åknes滑坡数据集加载器"""
    
    def __init__(self, config: Any):
        """
        初始化数据加载器
        
        Args:
            config: 配置对象，包含data_dir等参数
        """
        self.data_dir = Path(config.data.data_dir)
        self.start_date = "2010-01-01"
        self.end_date = "2023-03-27"
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.total_days = len(self.date_range)  # 应该是4834天
        
        # 定义19条序列的文件名映射
        self.sequence_files = {
            # 形变数据 (8条，新增KH0217)
            'GNSS_12H': 'GNSS_12H.csv',
            'KH0206_Displacement': 'KH0206_Displacement.csv',
            'KH0112_Displacement': 'KH0112_Displacement.csv',
            'KH0117_Displacement': 'KH0117_Displacement.csv',
            'KH0118_Displacement': 'KH0118_Displacement.csv',
            'KH0217_Displacement': 'KH0217_Displacement.csv',  # 新增
            'KH0218_Displacement': 'KH0218_Displacement.csv',
            'KH0306_Displacement': 'KH0306_Displacement.csv',
            
            # 孔隙水压数据 (7条，新增KH0217)
            'KH0206_Piezometer': 'KH0206_Piezometer.csv',
            'KH0112_Piezometer': 'KH0112_Piezometer.csv',
            'KH0117_Piezometer': 'KH0117_Piezometer.csv',
            'KH0118_Piezometer': 'KH0118_Piezometer.csv',
            'KH0217_Piezometer': 'KH0217_Piezometer.csv',  # 新增
            'KH0218_Piezometer': 'KH0218_Piezometer.csv',
            'KH0306_Piezometer': 'KH0306_Piezometer.csv',
            
            # 微震数据 (2条)
            'Seismicity_borehole': 'Seismicity_borehole.csv',
            'Seismicity_surface': 'Seismicity_surface.csv',
            
            # 气象数据 (4条)
            'Air_temperature': 'Air_temperature.csv',
            'Precipitation': 'Precipitation.csv',
            'Snow': 'Snow.csv',
            'Surface_runoff': 'Surface_runoff.csv'
        }
        
        # 序列分组用于可视化
        self.sequence_groups = {
            'deformation': ['GNSS_12H', 'KH0206_Displacement', 'KH0112_Displacement', 
                          'KH0117_Displacement', 'KH0118_Displacement', 'KH0217_Displacement',  # 新增
                          'KH0218_Displacement', 'KH0306_Displacement'],
            'piezometer': ['KH0206_Piezometer', 'KH0112_Piezometer', 'KH0117_Piezometer',
                          'KH0118_Piezometer', 'KH0217_Piezometer',  # 新增
                          'KH0218_Piezometer', 'KH0306_Piezometer'],
            'seismic': ['Seismicity_borehole', 'Seismicity_surface'],
            'meteorological': ['Air_temperature', 'Precipitation', 'Snow', 'Surface_runoff']
        }
    
    def load(self) -> pd.DataFrame:
        """
        加载所有19条时间序列数据
        
        Returns:
            pd.DataFrame: (4834, 19) 的DataFrame，index为DatetimeIndex，columns为19条序列名
        """
        logger.info("开始加载Åknes滑坡数据集...")
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame(index=self.date_range, columns=list(self.sequence_files.keys()))
        
        # 统计信息
        loading_stats = {}
        
        for seq_name, filename in self.sequence_files.items():
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}, 使用全NaN列占位")
                result_df[seq_name] = np.nan
                loading_stats[seq_name] = {
                    'data_points': 0,
                    'start_time': None,
                    'end_time': None,
                    'missing_rate': 1.0
                }
                continue
            
            try:
                # 根据序列类型选择不同的加载策略
                if seq_name == 'GNSS_12H':
                    series_data = self._load_gnss_data(file_path)
                elif seq_name.startswith('Seismicity'):
                    series_data = self._load_seismic_data(file_path)
                else:
                    series_data = self._load_standard_csv(file_path, seq_name)
                
                # 重采样到日尺度并对齐到统一日期索引
                if not series_data.empty:
                    # 确保索引是datetime类型
                    if not isinstance(series_data.index, pd.DatetimeIndex):
                        series_data.index = pd.to_datetime(series_data.index)
                    
                    # 重采样到日尺度（取均值）
                    daily_series = series_data.resample('D').mean()
                    
                    # 对齐到统一日期范围
                    aligned_series = daily_series.reindex(self.date_range)
                else:
                    aligned_series = pd.Series(np.nan, index=self.date_range)
                
                result_df[seq_name] = aligned_series
                
                # 计算统计信息
                data_points = len(series_data.dropna())
                start_time = series_data.index.min() if len(series_data) > 0 else None
                end_time = series_data.index.max() if len(series_data) > 0 else None
                missing_rate = aligned_series.isna().sum() / len(aligned_series)
                
                loading_stats[seq_name] = {
                    'data_points': data_points,
                    'start_time': start_time,
                    'end_time': end_time,
                    'missing_rate': missing_rate
                }
                
                logger.info(f"加载完成: {seq_name} - 数据点数: {data_points}, "
                          f"缺失率: {missing_rate:.2%}")
                
            except Exception as e:
                logger.error(f"加载 {seq_name} 时出错: {e}")
                result_df[seq_name] = np.nan
                loading_stats[seq_name] = {
                    'data_points': 0,
                    'start_time': None,
                    'end_time': None,
                    'missing_rate': 1.0
                }
        
        # 保存加载统计信息
        self.loading_stats = loading_stats
        
        logger.info(f"数据加载完成! DataFrame shape: {result_df.shape}")
        return result_df
    
    def _load_gnss_data(self, file_path: Path) -> pd.Series:
        """加载GNSS_12H数据，计算所有GPS点的平均三维合位移"""
        # 分块读取大文件以避免内存问题
        chunk_size = 10000
        all_displacements = []
        all_datetimes = []
        
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
            # 跳过标题行（如果存在）
            if 'instrument' in str(chunk.iloc[0, 0]):
                chunk = chunk.iloc[1:].copy()
            
            # 处理日期列
            chunk['datetime'] = pd.to_datetime(chunk['datetime'], format='mixed', dayfirst=False)
            
            # 计算每个时间点的三维合位移（欧几里得距离）
            displacement = np.sqrt(chunk['Northing']**2 + chunk['Easting']**2 + chunk['Height']**2)
            
            all_displacements.extend(displacement.values)
            all_datetimes.extend(chunk['datetime'].values)
        
        # 创建时间序列
        series = pd.Series(all_displacements, index=all_datetimes)
        
        # 按日期分组取平均值（因为同一时间可能有多个GPS点的数据）
        daily_avg = series.groupby(series.index).mean()
        
        return daily_avg
    
    def _load_seismic_data(self, file_path: Path) -> pd.Series:
        """加载微震数据，使用maxAmp或Amplitude作为振幅指标"""
        # 尝试不同的分隔符
        separators = [',', '\t', ';']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) >= 2:
                    break
            except Exception:
                continue
        
        if df is None:
            raise ValueError(f"无法读取微震文件 {file_path}")
        
        # 确定日期列和振幅列
        if 'borehole' in str(file_path).lower():
            # borehole文件通常使用制表符分隔
            date_col_candidates = ['Time']
            amplitude_col_candidates = ['maxAmp', 'geoMax', 'Vup', 'Vdown']
        else:  # surface
            date_col_candidates = ['Date']
            amplitude_col_candidates = ['Amplitude']
        
        # 找到实际存在的日期列
        date_col = None
        for col in date_col_candidates:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # 尝试所有列名中包含时间关键词的
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['time', 'date']):
                    date_col = col
                    break
        
        if date_col is None:
            logger.error(f"无法找到日期列，列名: {list(df.columns)}")
            return pd.Series(dtype=float)
        
        # 找到实际存在的振幅列（按优先级）
        amplitude_col = None
        if 'borehole' in str(file_path).lower():
            # borehole: 优先级 Vup, Vdown, maxAmp, geoMax
            amplitude_candidates = ['Vup', 'Vdown', 'maxAmp', 'geoMax']
        else:  # surface
            amplitude_candidates = ['Amplitude']
        
        for col in amplitude_candidates:
            if col in df.columns and df[col].notna().any():
                amplitude_col = col
                break
        
        if amplitude_col is None:
            # 尝试所有数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # 选择非空值最多的数值列
                non_null_counts = [(col, df[col].notna().sum()) for col in numeric_cols]
                non_null_counts.sort(key=lambda x: x[1], reverse=True)
                if non_null_counts and non_null_counts[0][1] > 0:
                    amplitude_col = non_null_counts[0][0]
                else:
                    amplitude_col = numeric_cols[0]
            else:
                logger.error(f"无法找到振幅列，列名: {list(df.columns)}")
                return pd.Series(dtype=float)
        
        # 处理日期
        try:
            df['datetime'] = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.warning(f"日期解析失败 {e}, 尝试其他格式")
            formats_to_try = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
            for fmt in formats_to_try:
                try:
                    df['datetime'] = pd.to_datetime(df[date_col], format=fmt)
                    break
                except:
                    continue
            else:
                df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 创建时间序列
        valid_mask = df['datetime'].notna() & df[amplitude_col].notna()
        if valid_mask.any():
            series = pd.Series(df.loc[valid_mask, amplitude_col].values, 
                             index=df.loc[valid_mask, 'datetime'])
        else:
            series = pd.Series(dtype=float)
        
        return series

    def _load_standard_csv(self, file_path: Path, seq_name: str) -> pd.Series:
        """加载标准CSV格式的数据"""
        # 尝试不同的分隔符
        separators = [',', '\t', ';']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) >= 2:  # 至少有日期和值两列
                    break
            except Exception:
                continue
        
        if df is None:
            raise ValueError(f"无法读取文件 {file_path}")
        
        # 确定日期列和值列
        date_col = None
        value_col = None
        
        # 查找日期列
        date_candidates = ['Date:', 'Date', 'datetime', 'Time']
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # 尝试第一列作为日期
            date_col = df.columns[0]
        
        # 查找值列（排除日期列和其他非数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            # 对于位移和孔隙水压数据，通常第二列或第三列是主要值
            if 'Displacement' in seq_name:
                # 寻找包含'Displacement'的列
                for col in df.columns:
                    if 'Displacement' in str(col):
                        value_col = col
                        break
                if value_col is None and len(numeric_cols) > 1:
                    value_col = numeric_cols[1]  # 第二个数值列
                elif value_col is None:
                    value_col = numeric_cols[0]
            elif 'Piezometer' in seq_name:
                # 寻找包含'Pressure'或类似关键词的列
                for col in df.columns:
                    if any(keyword in str(col).lower() for keyword in ['pressure', 'piezo', 'water']):
                        value_col = col
                        break
                if value_col is None and len(numeric_cols) > 1:
                    value_col = numeric_cols[1]
                elif value_col is None:
                    value_col = numeric_cols[0]
            else:
                # 其他情况使用第一个数值列
                value_col = numeric_cols[0]
        else:
            # 如果没有数值列，尝试转换所有列
            for col in df.columns:
                if col != date_col:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].notna().any():
                            value_col = col
                            break
                    except:
                        continue
        
        if value_col is None:
            raise ValueError(f"无法确定 {seq_name} 的值列")
        
        # 处理日期 - 移除弃用的infer_datetime_format参数
        try:
            df['datetime'] = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.warning(f"日期解析失败 {e}, 尝试欧洲格式")
            try:
                df['datetime'] = pd.to_datetime(df[date_col], format='%d-%m-%Y %H:%M:%S', errors='coerce')
                if df['datetime'].isna().all():
                    df['datetime'] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except:
                df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 创建时间序列
        valid_mask = df['datetime'].notna() & df[value_col].notna()
        if valid_mask.any():
            series = pd.Series(df.loc[valid_mask, value_col].values, 
                             index=df.loc[valid_mask, 'datetime'])
        else:
            series = pd.Series(dtype=float)
        
        return series
    
    def get_missing_report(self) -> Dict[str, Dict]:
        """
        获取每条序列的缺失统计
        
        Returns:
            Dict: 包含每条序列缺失统计的字典
        """
        if not hasattr(self, 'loading_stats'):
            raise ValueError("请先调用load()方法")
        
        missing_report = {}
        
        # 重新加载数据以进行详细缺失分析
        data_df = self.load()
        
        for seq_name in self.sequence_files.keys():
            series = data_df[seq_name]
            
            # 基本缺失统计
            total_missing = series.isna().sum()
            missing_rate = total_missing / len(series)
            
            # 最长连续缺失段
            na_groups = (~series.isna()).astype(int).groupby((~series.isna()).astype(int).diff().ne(0).cumsum()).sum()
            if len(na_groups) > 0:
                max_consecutive_missing = na_groups.max() if na_groups.max() > 0 else 0
            else:
                max_consecutive_missing = 0
            
            missing_report[seq_name] = {
                'total_missing_days': int(total_missing),
                'missing_rate': float(missing_rate),
                'max_consecutive_missing': int(max_consecutive_missing),
                'data_points': int(len(series) - total_missing)
            }
        
        return missing_report
    
    def plot_exploration(self, output_dir: str = "outputs/data_exploration"):
        """
        生成并保存探索性可视化
        
        Args:
            output_dir: 输出目录路径
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        data_df = self.load()
        
        # 1. 缺失值热力图 - 使用matplotlib替代seaborn
        plt.figure(figsize=(15, 8))
        missing_matrix = data_df.isna().T  # 转置以便序列在y轴
        # 使用imshow绘制热力图
        im = plt.imshow(missing_matrix.values, aspect='auto', cmap='binary_r', 
                       extent=[0, len(missing_matrix.columns), 0, len(missing_matrix.index)])
        plt.colorbar(im, label='Missing (White) / Present (Black)')
        
        # 设置标签
        x_ticks = np.arange(0, len(missing_matrix.columns), 100)
        if len(x_ticks) > 0:
            x_labels = [missing_matrix.columns[i].strftime('%Y-%m') for i in x_ticks]
            plt.xticks(x_ticks, x_labels, rotation=45)
        y_ticks = np.arange(0.5, len(missing_matrix.index), 1)
        y_labels = missing_matrix.index
        plt.yticks(y_ticks, y_labels)
        
        plt.title('Missing Data Heatmap (White = Missing, Black = Present)')
        plt.xlabel('Time')
        plt.ylabel('Time Series')
        plt.tight_layout()
        plt.savefig(output_path / 'missing_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 时间序列图（按组分面）
        fig, axes = plt.subplots(4, 1, figsize=(20, 16))
        fig.suptitle('Raw Time Series by Category', fontsize=16)
        
        group_colors = ['blue', 'red', 'green', 'orange']
        group_titles = ['Deformation', 'Piezometer', 'Seismic', 'Meteorological']
        
        for i, (group_name, sequences) in enumerate(self.sequence_groups.items()):
            ax = axes[i]
            for seq in sequences:
                if seq in data_df.columns:
                    series = data_df[seq].dropna()
                    if len(series) > 0:
                        ax.plot(series.index, series.values, label=seq, alpha=0.7)
            
            ax.set_title(group_titles[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'raw_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 基本统计量表格
        stats_df = data_df.describe()
        stats_df.loc['skew'] = data_df.skew()
        stats_df.loc['kurtosis'] = data_df.kurtosis()
        
        # 保存为CSV
        stats_df.to_csv(output_path / 'basic_statistics.csv')
        
        logger.info(f"探索性可视化已保存到: {output_path}")
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """获取各序列的基本统计量"""
        data_df = self.load()
        stats_df = data_df.describe()
        stats_df.loc['skew'] = data_df.skew()
        stats_df.loc['kurtosis'] = data_df.kurtosis()
        return stats_df


if __name__ == "__main__":
    # 测试代码
    class MockConfig:
        def __init__(self):
            self.data_dir = "/home/lab1111/zy/2024_AknesLandslide_Aspaas"
    
    config = MockConfig()
    loader = AknesDataLoader(config)
    
    # 加载数据
    data_df = loader.load()
    print(f"加载的数据形状: {data_df.shape}")
    print(f"日期范围: {data_df.index.min()} 到 {data_df.index.max()}")
    print(f"列名: {list(data_df.columns)}")
    
    # 获取缺失报告
    missing_report = loader.get_missing_report()
    print("\n缺失统计摘要:")
    for seq_name, stats in missing_report.items():
        print(f"{seq_name}: 缺失率={stats['missing_rate']:.2%}, "
              f"最长连续缺失={stats['max_consecutive_missing']}天")
    
    # 生成可视化（可选）
    # loader.plot_exploration()
    
    print("\n数据加载测试完成!")