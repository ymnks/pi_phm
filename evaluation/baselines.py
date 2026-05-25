#!/usr/bin/env python3
"""
增量预测基线实现
- Persistence增量版：未来7天每天增量 = 最近7天平均日增量  
- Linear增量版：用最近14天的日增量线性外推未来7天
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import pickle


def persistence_increment_baseline(
    gnss_series: pd.Series, 
    lookback_days: int = 60,
    forecast_days: int = 7,
    recent_window: int = 7
) -> np.ndarray:
    """
    Persistence增量基线
    
    Args:
        gnss_series: GNSS累计位移时间序列（物理空间，mm）
        lookback_days: 输入窗口长度
        forecast_days: 预测天数  
        recent_window: 用于计算平均日增量的最近天数
        
    Returns:
        预测的日增量数组，shape (forecast_days,)
    """
    # 计算日增量，处理NaN
    daily_increments = gnss_series.diff()
    daily_increments = daily_increments.fillna(0)
    
    # 取最近recent_window天的日增量
    recent_increments = daily_increments[-recent_window:]
    avg_daily_increment = recent_increments.mean()
    
    # 预测未来forecast_days天的日增量（每天都相同）
    predictions = np.full(forecast_days, avg_daily_increment)
    
    return predictions


def linear_increment_baseline(
    gnss_series: pd.Series,
    lookback_days: int = 60, 
    forecast_days: int = 7,
    fit_window: int = 14
) -> np.ndarray:
    """
    Linear增量基线 - 线性外推
    
    Args:
        gnss_series: GNSS累计位移时间序列（物理空间，mm）
        lookback_days: 输入窗口长度
        forecast_days: 预测天数
        fit_window: 用于线性拟合的最近天数
        
    Returns:
        预测的日增量数组，shape (forecast_days,)
    """
    # 计算日增量
    daily_increments = gnss_series.diff()
    daily_increments = daily_increments.fillna(0)
    
    # 取最近fit_window天的日增量和对应的时间索引
    recent_increments = daily_increments[-fit_window:]
    if len(recent_increments) < 2:
        # 如果数据不足，退化为persistence
        return persistence_increment_baseline(gnss_series, lookback_days, forecast_days)
    
    # 时间索引（以天为单位）
    time_indices = np.arange(len(recent_increments))
    
    # 线性拟合：increment = a * time + b
    coeffs = np.polyfit(time_indices, recent_increments.values, 1)
    a, b = coeffs[0], coeffs[1]
    
    # 预测未来forecast_days天的增量
    future_time_indices = np.arange(len(recent_increments), len(recent_increments) + forecast_days)
    predictions = a * future_time_indices + b
    
    return predictions


def evaluate_increment_baselines(
    df_features_norm: pd.DataFrame,
    normalizer_params_path: str,
    test_start_date: str,
    lookback_days: int = 60,
    forecast_days: int = 7
) -> dict:
    """
    在测试集上评估增量基线
    
    Args:
        df_features_norm: 归一化后的特征DataFrame
        normalizer_params_path: 归一化器参数文件路径
        test_start_date: 测试集开始日期
        lookback_days: 输入窗口长度
        forecast_days: 预测天数
        
    Returns:
        包含各基线评估指标的字典
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # 反归一化GNSS_12H
    with open(normalizer_params_path, 'rb') as f:
        normalizer_params = pickle.load(f)
    
    gnss_scaler = normalizer_params['scalers']['GNSS_12H']
    gnss_norm_values = df_features_norm['GNSS_12H'].values.reshape(-1, 1)
    gnss_raw_values = gnss_scaler.inverse_transform(gnss_norm_values).flatten()
    
    df_features_raw = df_features_norm.copy()
    df_features_raw['GNSS_12H'] = gnss_raw_values
    
    gnss_col = 'GNSS_12H'
    test_mask = df_features_raw.index >= pd.to_datetime(test_start_date)
    test_dates = df_features_raw[test_mask].index
    
    # 收集所有预测和真实值
    true_increments_list = []
    pers_preds_list = []
    linear_preds_list = []
    
    for date in test_dates:
        # 确保有足够的历史数据
        try:
            current_idx = df_features_raw.index.get_loc(date)
            start_idx = current_idx - lookback_days
            if start_idx < 0:
                continue
                
            # 获取输入窗口和目标窗口
            input_window = df_features_raw.iloc[start_idx:current_idx]
            # 目标是下一天的日增量
            if current_idx + 1 >= len(df_features_raw):
                continue
                
            target_value = df_features_raw.iloc[current_idx + 1][gnss_col]
            current_value = df_features_raw.iloc[current_idx][gnss_col]
            true_increment = target_value - current_value
            
            # 基线预测（1天预测）
            pers_pred_single = persistence_increment_baseline(
                input_window[gnss_col], lookback_days, 1
            )[0]
            linear_pred_single = linear_increment_baseline(
                input_window[gnss_col], lookback_days, 1
            )[0]
            
            true_increments_list.append(true_increment)
            pers_preds_list.append(pers_pred_single)
            linear_preds_list.append(linear_pred_single)
            
        except Exception as e:
            continue  # 跳过有问题的样本
    
    if len(true_increments_list) == 0:
        raise ValueError("No valid test samples found")
    
    # 转换为数组
    true_increments_all = np.array(true_increments_list)
    pers_preds_all = np.array(pers_preds_list)
    linear_preds_all = np.array(linear_preds_list)
    
    # 过滤NaN值
    valid_mask = ~(np.isnan(true_increments_all) | np.isnan(pers_preds_all) | np.isnan(linear_preds_all))
    true_increments_all = true_increments_all[valid_mask]
    pers_preds_all = pers_preds_all[valid_mask]
    linear_preds_all = linear_preds_all[valid_mask]
    
    if len(true_increments_all) == 0:
        raise ValueError("No valid samples after NaN filtering")
    
    # 计算指标
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    pers_metrics = compute_metrics(true_increments_all, pers_preds_all)
    linear_metrics = compute_metrics(true_increments_all, linear_preds_all)
    
    return {
        'persistence_increment': pers_metrics,
        'linear_increment': linear_metrics,
        'sample_count': len(true_increments_all),
        'true_increments': true_increments_all,
        'pers_predictions': pers_preds_all,
        'linear_predictions': linear_preds_all
    }