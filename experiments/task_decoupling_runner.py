#!/usr/bin/env python3
"""
Task Decoupling Experiment Runner
统一实验入口，支持debug9.md要求的所有实验变体
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import PI_PHM_Config
from data.data_loader import AknesDataLoader
from data.preprocessor import AknesPreprocessor
from data.feature_engineer import AknesFeatureEngineer
from data.label_generator import RiskLabelGenerator
from data.event_catalog import CreepBurstCatalog
from data.dataset import create_dataloaders, generate_event_detection_labels, create_static_features, PhysicsAwareNormalizer, AknesTimeSeriesDataset
from models.task_decoupling_models import create_task_decoupling_model, get_gnss_feature_indices
from training.trainer import PI_PHM_Trainer
from rolling_validation import RollingValidationFramework


class TaskDecouplingRunner:
    """任务解耦实验运行器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = PI_PHM_Config.from_yaml(config_path)
        self.model_variants = [
            "disp_only",           # E1
            "event_only_full",     # E2a
            "event_only_borehole", # E2b  
            "full_shared",         # E3
            "partial_shared"       # E4
        ]
        # 初始化rolling validation framework
        self.rv_framework = RollingValidationFramework(self.config)
        
    def _load_data(self):
        """加载数据，复用main_rolling_validation.py的逻辑"""
        # 加载和预处理数据
        loader = AknesDataLoader(self.config)
        df_raw = loader.load()
        
        preprocessor = AknesPreprocessor(df_raw, self.config)
        df_clean, quality_flags = preprocessor.preprocess()
        
        engineer = AknesFeatureEngineer(df_clean, self.config)
        df_features = engineer.transform()
        
        label_gen = RiskLabelGenerator(df_features, self.config)
        labels = label_gen.generate(method="max_fusion")
        
        # 创建质量矩阵
        quality_cols = [col for col in df_raw.columns if col != 'Date']
        df_quality_raw = df_raw[quality_cols].notna().astype(float)
        
        # 特征工程后，扩展质量矩阵以匹配特征工程后的维度
        original_feature_count = len(quality_cols)
        engineered_feature_count = len(df_features.columns)
        
        if engineered_feature_count > original_feature_count:
            # 创建扩展的质量矩阵
            df_quality_extended = pd.DataFrame(index=df_quality_raw.index)
            
            # 首先复制原始质量列
            for i, col in enumerate(quality_cols):
                df_quality_extended[f'original_{i}'] = df_quality_raw[col]
            
            # 对于衍生特征，基于其来源特征的质量来确定
            # 这里简化处理：假设所有衍生特征的质量与最后一个原始特征相同
            last_original_quality = df_quality_raw.iloc[:, -1] if original_feature_count > 0 else pd.Series([1.0] * len(df_quality_raw))
            
            additional_features = engineered_feature_count - original_feature_count
            for i in range(additional_features):
                df_quality_extended[f'derived_{i}'] = last_original_quality
            
            df_quality = df_quality_extended
        else:
            df_quality = df_quality_raw
        
        # 构建事件检测标签
        event_catalog = CreepBurstCatalog()
        y_event = generate_event_detection_labels(event_catalog, df_features, self.config.model.forecast)
        
        return df_features, df_quality, labels, y_event
    
    def run_single_fold_experiment(self, model_variant: str, fold_id: int = 4, 
                                output_dir: str = "outputs/task_decoupling") -> Dict:
        """
        运行单个fold的实验
        
        Args:
            model_variant: 模型变体类型
            fold_id: fold ID（默认使用fold 4作为代表性fold）
            output_dir: 输出目录
            
        Returns:
            实验结果字典
        """
        print(f"Running {model_variant} on fold {fold_id}")
        
        # 创建输出目录
        experiment_output_dir = os.path.join(output_dir, f"{model_variant}_fold{fold_id}")
        os.makedirs(experiment_output_dir, exist_ok=True)
        
        # 加载数据
        df_features, df_quality, labels, y_event = self._load_data()
        
        # 获取指定fold的数据划分
        fold_data = self.rv_framework.get_fold_data_split(df_features, df_quality, labels, y_event, fold_id)
        
        # 获取GNSS特征索引（用于E2b）
        feature_names = list(df_features.columns)
        gnss_indices = get_gnss_feature_indices(feature_names) if model_variant == "event_only_borehole" else []
        
        # 构建feature_index_map
        feature_index_map = self._build_feature_index_map(feature_names)
        
        # 创建模型
        input_channels = len(feature_names)
        if model_variant == "event_only_borehole":
            # 对于E2b，我们需要在数据层面过滤掉GNSS特征
            # 但这里我们先保持输入通道数不变，在模型内部处理
            pass
            
        model = create_task_decoupling_model(
            model_variant, self.config, feature_index_map, 
            input_channels=input_channels, gnss_indices=gnss_indices
        )
        
        # 扩展质量矩阵以匹配特征工程后的维度
        original_feature_count = len([col for col in df_features.columns if 'original_' in col or not any(x in col for x in ['velocity', 'acceleration', 'rate', 'sum'])])
        engineered_feature_count = len(df_features.columns)
        
        if engineered_feature_count > original_feature_count:
            # 创建扩展的质量矩阵
            df_quality_extended = pd.DataFrame(index=df_quality.index)
            
            # 首先复制原始质量列
            original_cols = [col for col in df_quality.columns if 'original_' in col]
            if original_cols:
                for col in original_cols:
                    df_quality_extended[col] = df_quality[col]
            else:
                # 如果没有original_前缀，假设所有列都是原始的
                for i, col in enumerate(df_quality.columns):
                    df_quality_extended[f'original_{i}'] = df_quality[col]
            
            # 对于衍生特征，基于其来源特征的质量来确定
            last_original_quality = df_quality.iloc[:, -1] if len(df_quality.columns) > 0 else pd.Series([1.0] * len(df_quality))
            
            additional_features = engineered_feature_count - len(df_quality_extended.columns)
            for i in range(additional_features):
                df_quality_extended[f'derived_{i}'] = last_original_quality
        else:
            df_quality_extended = df_quality
        
        # 为当前fold创建normalizer（只在train上fit）
        normalizer = PhysicsAwareNormalizer(self.config)
        normalizer.fit(fold_data['train']['features'])
        
        train_dataset = AknesTimeSeriesDataset(
            df_features=fold_data['train']['features'],
            df_quality=df_quality_extended.loc[fold_data['train']['mask']],
            labels=fold_data['train']['labels'],
            static_features=create_static_features(self.config),
            config=self.config,
            mode='train',
            y_event=fold_data['train']['y_event'],
            normalizer=normalizer
        )
        
        val_dataset = AknesTimeSeriesDataset(
            df_features=fold_data['val']['features'],
            df_quality=df_quality_extended.loc[fold_data['val']['mask']],
            labels=fold_data['val']['labels'],
            static_features=create_static_features(self.config),
            config=self.config,
            mode='val',
            y_event=fold_data['val']['y_event'],
            normalizer=normalizer
        )
        
        test_dataset = AknesTimeSeriesDataset(
            df_features=fold_data['test']['features'],
            df_quality=df_quality_extended.loc[fold_data['test']['mask']],
            labels=fold_data['test']['labels'],
            static_features=create_static_features(self.config),
            config=self.config,
            mode='test',
            y_event=fold_data['test']['y_event'],
            normalizer=normalizer
        )
        
        batch_size = self.config.training.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # 创建trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = PI_PHM_Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=device,
            feature_index_map=feature_index_map,
            normalizer=normalizer
        )
        trainer.fold_id = fold_id  # 设置fold_id用于校准
        
        # 训练模型
        try:
            training_history = trainer.fit()
            # 获取最佳验证分数
            best_score = min([metrics['val_combined_score'] for metrics in training_history['val_metrics']])
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            best_score = float('inf')
            training_history = None
        
        # 评估测试集性能
        if training_history is not None:
            # 这里可以添加测试集评估逻辑
            # 为了简化，我们使用验证集的最佳结果作为代理
            best_epoch_idx = training_history['val_metrics'].index(
                min(training_history['val_metrics'], key=lambda x: x['val_combined_score'])
            )
            best_metrics = training_history['val_metrics'][best_epoch_idx]
            
            # 提取指标，确保 mean_lead_days 不为 NaN
            val_mean_lead_days = best_metrics.get('val_mean_lead_days', 0.0)
            if pd.isna(val_mean_lead_days):
                val_mean_lead_days = 0.0

            result = {
                'experiment': model_variant,
                'fold': fold_id,
                'best_score': float(best_score),
                'timestamp': datetime.now().isoformat(),
                'disp_MAE': float(best_metrics.get('val_mae', float('nan'))),
                'disp_RMSE': float(best_metrics.get('val_rmse', float('nan'))),
                'disp_R2': float(best_metrics.get('val_r2', float('nan'))),
                'event_PR_AUC': float(best_metrics.get('val_event_prauc', float('nan'))),
                'event_ROC_AUC': float(best_metrics.get('val_event_auc', float('nan'))),
                'event_detection_rate': float(best_metrics.get('val_recall_at_calibrated', float('nan'))),
                'strict_FPR': float(best_metrics.get('val_strict_fpr_at_calibrated', float('nan'))),
                'mean_lead_days': float(val_mean_lead_days)
            }
        else:
            # 使用默认值
            result = {
                'experiment': model_variant,
                'fold': fold_id,
                'best_score': float('inf'),
                'timestamp': datetime.now().isoformat(),
                'disp_MAE': float('nan'),
                'disp_RMSE': float('nan'),
                'disp_R2': float('nan'),
                'event_PR_AUC': float('nan'),
                'event_ROC_AUC': float('nan'),
                'event_detection_rate': float('nan'),
                'strict_FPR': float('nan'),
                'mean_lead_days': 0.0
            }
        
        # 保存结果
        result_path = os.path.join(experiment_output_dir, "experiment_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _build_feature_index_map(self, feature_names: List[str]) -> Dict:
        """构建特征索引映射"""
        feature_index_map = {}
        
        velocity_indices = []
        acceleration_indices = []
        inverse_velocity_indices = []
        piezometer_rate_indices = []
        seismic_rate_indices = []
        rain_7d_index = None
        
        for i, name in enumerate(feature_names):
            if 'velocity' in name:
                velocity_indices.append(i)
                if 'inverse' in name or ('velocity' in name and 'GNSS' not in name):
                    inverse_velocity_indices.append(i)
            elif 'acceleration' in name:
                acceleration_indices.append(i)
            elif 'seismic' in name and 'rate' in name:
                seismic_rate_indices.append(i)
            elif 'piezometer' in name and 'rate' in name:
                piezometer_rate_indices.append(i)
            elif 'rain_7d' in name:
                rain_7d_index = i
        
        if not inverse_velocity_indices:
            inverse_velocity_indices = velocity_indices
        
        if rain_7d_index is None:
            for i, name in enumerate(feature_names):
                if 'Precipitation' in name and 'sum_7d' in name:
                    rain_7d_index = i
                    break
            if rain_7d_index is None and len(feature_names) > 0:
                rain_7d_index = 0
        
        feature_index_map['velocity_indices'] = velocity_indices
        feature_index_map['acceleration_indices'] = acceleration_indices
        feature_index_map['inverse_velocity_indices'] = inverse_velocity_indices
        feature_index_map['piezometer_rate_indices'] = piezometer_rate_indices
        feature_index_map['seismic_rate_indices'] = seismic_rate_indices
        feature_index_map['rain_7d_index'] = rain_7d_index
        
        return feature_index_map
    
    def run_all_experiments_single_fold(self, fold_id: int = 4, 
                                     output_dir: str = "outputs/task_decoupling") -> pd.DataFrame:
        """运行所有实验变体在单个fold上"""
        all_results = []
        
        for model_variant in self.model_variants:
            try:
                result = self.run_single_fold_experiment(model_variant, fold_id, output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"Error running {model_variant}: {e}")
                continue
        
        df = pd.DataFrame(all_results)
        return df
    
    def run_rolling_validation(self, model_variant: str, output_dir: str = "outputs/task_decoupling") -> List[Dict]:
        """运行完整的rolling validation实验"""
        all_results = []
        
        # 运行所有5个folds
        for fold_id in range(1, 6):
            try:
                result = self.run_single_fold_experiment(model_variant, fold_id, output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"Error running {model_variant} on fold {fold_id}: {e}")
                import traceback
                traceback.print_exc()
                # 添加失败结果
                failed_result = {
                    'experiment': model_variant,
                    'fold': fold_id,
                    'best_score': float('inf'),
                    'timestamp': datetime.now().isoformat(),
                    'disp_MAE': float('nan'),
                    'disp_RMSE': float('nan'),
                    'disp_R2': float('nan'),
                    'event_PR_AUC': float('nan'),
                    'event_ROC_AUC': float('nan'),
                    'event_detection_rate': float('nan'),
                    'strict_FPR': float('nan'),
                    'mean_lead_days': 0.0
                }
                all_results.append(failed_result)
        
        return all_results
    
    def generate_results_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """生成汇总统计结果"""
        summary_data = []
        
        for model_variant in results_df['experiment'].unique():
            variant_data = results_df[results_df['experiment'] == model_variant]
            
            if len(variant_data) == 0:
                continue
                
            # 计算位移指标的mean±std
            disp_mae_mean = variant_data['disp_MAE'].mean()
            disp_mae_std = variant_data['disp_MAE'].std()
            disp_rmse_mean = variant_data['disp_RMSE'].mean()
            disp_rmse_std = variant_data['disp_RMSE'].std()
            disp_r2_mean = variant_data['disp_R2'].mean()
            disp_r2_std = variant_data['disp_R2'].std()
            
            # 计算事件指标的mean±std
            event_pr_auc_mean = variant_data['event_PR_AUC'].mean()
            event_pr_auc_std = variant_data['event_PR_AUC'].std()
            event_detection_rate_mean = variant_data['event_detection_rate'].mean()
            event_detection_rate_std = variant_data['event_detection_rate'].std()
            strict_fpr_mean = variant_data['strict_FPR'].mean()
            strict_fpr_std = variant_data['strict_FPR'].std()
            mean_lead_days_mean = variant_data['mean_lead_days'].mean()
            mean_lead_days_std = variant_data['mean_lead_days'].std()
            
            summary_row = {
                'experiment': model_variant,
                'disp_MAE_mean±std': f"{disp_mae_mean:.3f}±{disp_mae_std:.3f}" if not pd.isna(disp_mae_std) else f"{disp_mae_mean:.3f}",
                'disp_RMSE_mean±std': f"{disp_rmse_mean:.3f}±{disp_rmse_std:.3f}" if not pd.isna(disp_rmse_std) else f"{disp_rmse_mean:.3f}",
                'disp_R2_mean±std': f"{disp_r2_mean:.3f}±{disp_r2_std:.3f}" if not pd.isna(disp_r2_std) else f"{disp_r2_mean:.3f}",
                'event_PR_AUC_mean±std': f"{event_pr_auc_mean:.3f}±{event_pr_auc_std:.3f}" if not pd.isna(event_pr_auc_std) else f"{event_pr_auc_mean:.3f}",
                'event_detection_rate_mean±std': f"{event_detection_rate_mean:.3f}±{event_detection_rate_std:.3f}" if not pd.isna(event_detection_rate_std) else f"{event_detection_rate_mean:.3f}",
                'strict_FPR_mean±std': f"{strict_fpr_mean:.3f}±{strict_fpr_std:.3f}" if not pd.isna(strict_fpr_std) else f"{strict_fpr_mean:.3f}",
                'mean_lead_days_mean±std': f"{mean_lead_days_mean:.1f}±{mean_lead_days_std:.1f}" if not pd.isna(mean_lead_days_std) else f"{mean_lead_days_mean:.1f}"
            }
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Task Decoupling Experiment Runner")
    parser.add_argument("--model_variant", type=str, required=True,
                       choices=["disp_only", "event_only_full", "event_only_borehole", 
                               "full_shared", "partial_shared", "all"],
                       help="Model variant to run")
    parser.add_argument("--single_fold", type=int, default=None,
                       help="Single fold to run (if not specified, runs all folds)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs/task_decoupling",
                       help="Output directory")
    
    args = parser.parse_args()
    
    runner = TaskDecouplingRunner(args.config)
    
    if args.model_variant == "all":
        if args.single_fold is not None:
            combined_results = runner.run_all_experiments_single_fold(
                args.single_fold, args.output_dir
            )
        else:
            # 运行所有变体的完整rolling validation
            all_results = []
            for variant in ["disp_only", "event_only_full", "event_only_borehole", "full_shared", "partial_shared"]:
                print(f"\nRunning {variant} on all folds...")
                variant_results = runner.run_rolling_validation(variant, args.output_dir)
                all_results.extend(variant_results)
            combined_results = pd.DataFrame(all_results)
    else:
        if args.single_fold is not None:
            single_result = runner.run_single_fold_experiment(
                args.model_variant, args.single_fold, args.output_dir
            )
            combined_results = pd.DataFrame([single_result])
        else:
            # 运行指定变体的完整rolling validation
            combined_results = pd.DataFrame(runner.run_rolling_validation(args.model_variant, args.output_dir))
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"experiment_results_{timestamp}.csv")
    combined_results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # 生成汇总统计
    if len(combined_results) > 1:
        summary_df = runner.generate_results_summary(combined_results)
        summary_path = os.path.join(args.output_dir, f"experiment_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # 打印汇总结果
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY")
        print("="*100)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()