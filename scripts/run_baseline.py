#!/usr/bin/env python3
"""
PI-PHM 基准模型脚本
提供2个简单基准模型的训练和评估：
- Baseline 1: 3层 LSTM + Linear（相同的输入特征和训练划分）
- Baseline 2: 简单 Linear Regression（持久模型：预测值=最后一天的值重复7天）
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import create_dataloaders, PhysicsAwareNormalizer
from evaluation.evaluator import PIPHMEvaluator


def setup_logging():
    """设置日志记录"""
    os.makedirs('outputs/baselines', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/baselines/baseline_run.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class LSTMBaseline(nn.Module):
    """LSTM基准模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


def train_lstm_baseline(config, logger, device):
    """训练LSTM基准模型"""
    logger.info("Training LSTM baseline model...")
    
    # 加载预处理数据
    data_dir = 'outputs/data'
    df_features_norm = pd.read_parquet(f'{data_dir}/features_normalized.parquet')
    labels = pd.read_csv(f'{data_dir}/risk_labels.csv', index_col=0, parse_dates=True)
    df_quality = pd.read_parquet(f'{data_dir}/quality_flags.parquet')
    normalizer = PhysicsAwareNormalizer(config)
    normalizer.load('outputs/normalizer_params.pkl')
    
    # 创建数据集
    train_loader, val_loader, test_loader = create_dataloaders(
        df_features_norm, df_quality, labels, config
    )
    
    # 创建模型
    input_size = df_features_norm.shape[1]
    model = LSTMBaseline(input_size, hidden_size=128, num_layers=3, output_size=7)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(50):  # 较短的训练周期
        # 训练
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['x_dynamic'].float().to(device)
            y = batch['y_disp_main'].float().to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x_dynamic'].float().to(device)
                y = batch['y_disp_main'].float().to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'outputs/baselines/lstm_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('outputs/baselines/lstm_best.pt'))
    
    # 测试评估
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_dynamic'].float().to(device)
            y = batch['y_disp_main'].float().to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # 反归一化
    gnss_col_idx = df_features_norm.columns.get_loc('GNSS_12H')
    all_preds_denorm = normalizer.inverse_transform_single_feature(all_preds, gnss_col_idx)
    all_trues_denorm = normalizer.inverse_transform_single_feature(all_trues, gnss_col_idx)
    
    # 计算指标
    mae = mean_absolute_error(all_trues_denorm, all_preds_denorm)
    rmse = np.sqrt(mean_squared_error(all_trues_denorm, all_preds_denorm))
    r2 = r2_score(all_trues_denorm, all_preds_denorm)
    
    logger.info(f"LSTM Baseline Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def evaluate_persistence_baseline(config, logger):
    """评估持久性基准模型（预测值=最后一天的值重复7天）"""
    logger.info("Evaluating persistence baseline model...")
    
    # 加载预处理数据
    data_dir = 'outputs/data'
    df_features_norm = pd.read_parquet(f'{data_dir}/features_normalized.parquet')
    labels = pd.read_csv(f'{data_dir}/risk_labels.csv', index_col=0, parse_dates=True)
    normalizer = PhysicsAwareNormalizer(config)
    normalizer.load('outputs/normalizer_params.pkl')
    
    # 获取GNSS_12H列
    gnss_col_idx = df_features_norm.columns.get_loc('GNSS_12H')
    gnss_series = df_features_norm.iloc[:, gnss_col_idx].values
    
    # 创建测试集预测
    lookback = config.lookback
    forecast = config.forecast
    test_start_idx = len(df_features_norm) - len(labels[(labels.index > pd.to_datetime(config.test_start_date))])
    
    preds = []
    trues = []
    
    for i in range(test_start_idx, len(df_features_norm) - forecast + 1):
        if i + lookback + forecast <= len(df_features_norm):
            # 最后一个观测值作为未来7天的预测
            last_obs = gnss_series[i + lookback - 1]
            pred = np.full(forecast, last_obs)
            true = gnss_series[i + lookback:i + lookback + forecast]
            preds.append(pred)
            trues.append(true)
    
    preds = np.array(preds)
    trues = np.array(trues)
    
    # 反归一化
    preds_denorm = normalizer.inverse_transform_single_feature(preds, gnss_col_idx)
    trues_denorm = normalizer.inverse_transform_single_feature(trues, gnss_col_idx)
    
    # 计算指标
    mae = mean_absolute_error(trues_denorm.flatten(), preds_denorm.flatten())
    rmse = np.sqrt(mean_squared_error(trues_denorm.flatten(), preds_denorm.flatten()))
    r2 = r2_score(trues_denorm.flatten(), preds_denorm.flatten())
    
    logger.info(f"Persistence Baseline Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description='PI-PHM Baseline Models')
    parser.add_argument('--baseline', type=str, default='all',
                       choices=['lstm', 'persistence', 'all'],
                       help='Which baseline to run')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 加载配置
    config = Config()
    if args.config:
        config.load_from_yaml(args.config)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    results = {}
    
    try:
        if args.baseline in ['lstm', 'all']:
            results['lstm'] = train_lstm_baseline(config, logger, device)
        
        if args.baseline in ['persistence', 'all']:
            results['persistence'] = evaluate_persistence_baseline(config, logger)
        
        # 保存结果
        with open('outputs/baselines/baseline_results.txt', 'w') as f:
            f.write("Baseline Model Results\n")
            f.write("=" * 40 + "\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info("Baseline evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()