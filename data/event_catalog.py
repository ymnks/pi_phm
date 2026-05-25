#!/usr/bin/env python3
"""
蠕变爆发事件目录 - 基于Aspaas et al. (2024)论文Table S2的真实事件数据
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CreepBurstEvent:
    """蠕变爆发事件数据结构"""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    boreholes: List[str]       # 涉及哪些钻孔
    displacement_mm: float     # 总位移量
    duration_days: float       # 持续时间
    max_velocity_mm_day: float # 最大速度
    severity: str              # 自动计算：minor/moderate/major

class CreepBurstCatalog:
    """蠕变爆发事件目录管理器"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化蠕变爆发事件目录
        
        Args:
            csv_path: Table S2 CSV文件路径，如果为None则使用内置数据
        """
        self.events: List[CreepBurstEvent] = []
        self.borehole_mapping = {
            'KH-02-18': 'KH0218_Displacement',
            'KH-01-18': 'KH0118_Displacement', 
            'KH-01-17': 'KH0117_Displacement',
            'KH-02-17 Upper': 'KH0217_Displacement',
            'KH-02-17 Lower': 'KH0217_Displacement',
            'KH-01-12 Upper': 'KH0112_Displacement',
            'KH-01-12 Lower': 'KH0112_Displacement',
            'KH-02-06 Upper': 'KH0206_Displacement',
            'KH-02-06 Lower': 'KH0206_Displacement'
        }
        
        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            # 如果没有提供CSV路径，尝试从默认位置加载
            default_path = "/home/lab1111/zy/2024_AknesLandslide_Aspaas/Table_S2.csv"
            if Path(default_path).exists():
                self._load_from_csv(default_path)
            else:
                logger.warning(f"Default CSV path {default_path} not found, catalog will be empty")
    
    def _load_from_csv(self, csv_path: str):
        """从CSV文件加载蠕变爆发事件"""
        logger.info(f"Loading creep burst events from {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path, sep='\t', skiprows=1, encoding='utf-8')
        
        # 处理列名
        columns = [
            'Start time', 'End time', 'KH-02-18', 'KH-01-18', 'KH-01-17',
            'KH-02-17 Upper', 'KH-02-17 Lower', 'KH-01-12 Upper', 'KH-01-12 Lower',
            'KH-02-06 Upper', 'KH-02-06 Lower', 'Displacement (mm)', 'Duration (days)', 'Max velocity (mm/day)'
        ]
        
        # 确保列名正确
        df.columns = columns[:len(df.columns)]
        
        # 转换时间格式（处理欧洲日期格式 DD.MM.YYYY HH:MM）
        df['Start time'] = pd.to_datetime(df['Start time'], format='%d.%m.%Y %H:%M')
        df['End time'] = pd.to_datetime(df['End time'], format='%d.%m.%Y %H:%M')
        
        # 处理数值列（将逗号替换为点，并转换为float）
        numeric_columns = ['Displacement (mm)', 'Duration (days)', 'Max velocity (mm/day)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # 解析每个事件
        borehole_cols = list(self.borehole_mapping.keys())
        
        for idx, row in df.iterrows():
            # 获取涉及的钻孔
            active_boreholes = []
            for borehole_col in borehole_cols:
                if borehole_col in row and pd.notna(row[borehole_col]) and str(row[borehole_col]).strip() != '':
                    # 非空值表示该钻孔在此事件中活跃
                    active_boreholes.append(borehole_col)
            
            # 跳过没有活跃钻孔的事件
            if not active_boreholes:
                continue
            
            # 获取事件参数
            displacement = row['Displacement (mm)'] if pd.notna(row['Displacement (mm)']) else 0.0
            duration = row['Duration (days)'] if pd.notna(row['Duration (days)']) else 1.0
            max_velocity = row['Max velocity (mm/day)'] if pd.notna(row['Max velocity (mm/day)']) else 0.0
            
            # 计算严重程度
            severity = self._calculate_severity(displacement, max_velocity)
            
            # 创建事件对象
            event = CreepBurstEvent(
                start_time=row['Start time'],
                end_time=row['End time'],
                boreholes=active_boreholes,
                displacement_mm=displacement,
                duration_days=duration,
                max_velocity_mm_day=max_velocity,
                severity=severity
            )
            
            self.events.append(event)
        
        logger.info(f"Loaded {len(self.events)} creep burst events")
    
    def _calculate_severity(self, displacement: float, max_velocity: float) -> str:
        """
        根据位移量和最大速度计算事件严重程度
        
        severity分级规则（基于论文统计）：
          minor:    displacement < 0.5mm 且 max_velocity < 2.0 mm/day
          moderate: 0.5mm <= displacement < 1.5mm 或 2.0 <= max_velocity < 5.0 mm/day  
          major:    displacement >= 1.5mm 或 max_velocity >= 5.0 mm/day
        """
        if displacement < 0.5 and max_velocity < 2.0:
            return "minor"
        elif (0.5 <= displacement < 1.5) or (2.0 <= max_velocity < 5.0):
            return "moderate"
        else:
            return "major"
    
    def get_events_in_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[CreepBurstEvent]:
        """获取指定日期范围内的事件"""
        filtered_events = []
        for event in self.events:
            if event.start_time <= end_date and event.end_time >= start_date:
                filtered_events.append(event)
        return filtered_events
    
    def get_sync_events(self, date: pd.Timestamp, window_days: int = 3) -> List[CreepBurstEvent]:
        """
        获取指定日期前后window_days天内的同步事件
        
        Args:
            date: 目标日期
            window_days: 时间窗口天数（前后各window_days天）
            
        Returns:
            在时间窗口内发生的事件列表
        """
        start_window = date - pd.Timedelta(days=window_days)
        end_window = date + pd.Timedelta(days=window_days)
        return self.get_events_in_date_range(start_window, end_window)
    
    def count_sync_boreholes(self, date: pd.Timestamp, window_days: int = 3) -> int:
        """
        计算指定日期前后window_days天内活跃的钻孔数量（去重）
        
        Args:
            date: 目标日期
            window_days: 时间窗口天数
            
        Returns:
            活跃钻孔数量
        """
        sync_events = self.get_sync_events(date, window_days)
        all_boreholes = set()
        for event in sync_events:
            all_boreholes.update(event.boreholes)
        return len(all_boreholes)
    
    def get_statistics(self) -> Dict:
        """获取事件统计信息"""
        if not self.events:
            return {}
        
        displacements = [e.displacement_mm for e in self.events]
        durations = [e.duration_days for e in self.events]
        velocities = [e.max_velocity_mm_day for e in self.events]
        severities = [e.severity for e in self.events]
        
        return {
            'total_events': len(self.events),
            'displacement_stats': {
                'min': min(displacements),
                'max': max(displacements),
                'median': np.median(displacements),
                'mean': np.mean(displacements)
            },
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'median': np.median(durations),
                'mean': np.mean(durations)
            },
            'velocity_stats': {
                'min': min(velocities),
                'max': max(velocities),
                'median': np.median(velocities),
                'mean': np.mean(velocities)
            },
            'severity_counts': {
                'minor': severities.count('minor'),
                'moderate': severities.count('moderate'),
                'major': severities.count('major')
            }
        }
    
    def print_summary(self):
        """打印事件摘要"""
        stats = self.get_statistics()
        if not stats:
            logger.info("No events loaded")
            return
        
        print(f"蠕变爆发事件统计摘要:")
        print(f"  总事件数: {stats['total_events']}")
        print(f"  位移量: min={stats['displacement_stats']['min']:.2f}mm, "
              f"max={stats['displacement_stats']['max']:.2f}mm, "
              f"median={stats['displacement_stats']['median']:.2f}mm")
        print(f"  持续时间: min={stats['duration_stats']['min']:.1f}天, "
              f"max={stats['duration_stats']['max']:.1f}天, "
              f"median={stats['duration_stats']['median']:.1f}天")
        print(f"  最大速度: min={stats['velocity_stats']['min']:.2f}mm/day, "
              f"max={stats['velocity_stats']['max']:.2f}mm/day, "
              f"median={stats['velocity_stats']['median']:.2f}mm/day")
        print(f"  严重程度分布: minor={stats['severity_counts']['minor']}, "
              f"moderate={stats['severity_counts']['moderate']}, "
              f"major={stats['severity_counts']['major']}")