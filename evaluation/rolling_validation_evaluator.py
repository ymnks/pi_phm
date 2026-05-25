#!/usr/bin/env python3
"""
Rolling Validation Evaluator for PI-PHM
实现P4任务要求的评估输出
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluator import PIPHMEvaluator
from training.checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class RollingValidationEvaluator:
    """Rolling Validation Evaluator for P4 task"""
    
    def __init__(self, config, output_dir: str = "outputs/rolling_validation"):
        self.config = config
        self.output_dir = output_dir
        self.evaluator = PIPHMEvaluator(config)
        
    def evaluate_fold_checkpoints(self, fold_id: int, val_loader, normalizer, device='cpu') -> List[Dict]:
        """
        评估指定fold的4类checkpoint
        
        Args:
            fold_id: fold ID
            val_loader: 验证数据加载器
            normalizer: 归一化器
            device: 设备
            
        Returns:
            List[Dict]: 包含4个checkpoint评估结果的列表
        """
        checkpoint_types = ['best_disp', 'best_event', 'best_multi', 'last']
        results = []
        
        for checkpoint_type in checkpoint_types:
            try:
                # 构建checkpoint路径
                checkpoint_filename = f"{checkpoint_type}_fold{fold_id}"
                checkpoint_path = os.path.join("outputs/checkpoints", f"{checkpoint_filename}.pth")
                metadata_path = os.path.join("outputs/checkpoints", f"{checkpoint_filename}_metadata.json")
                
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint {checkpoint_path} not found, skipping")
                    continue
                
                # 加载checkpoint和metadata
                checkpoint = torch.load(checkpoint_path, map_location=device)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # 加载模型
                model = self._load_model_from_checkpoint(checkpoint)
                
                # 获取校准阈值
                calibrated_thresholds = None
                if 'val_threshold_f2_best' in metadata:
                    calibrated_thresholds = {
                        'threshold_f2_best': metadata['val_threshold_f2_best'],
                        'threshold_strict': metadata['val_threshold_strict'],
                        'threshold_loose': metadata['val_threshold_loose']
                    }
                
                # 评估模型
                eval_result = self.evaluator.evaluate(
                    model, val_loader, normalizer, device, calibrated_thresholds
                )
                
                metrics = eval_result['metrics']
                
                # 提取所需指标
                result = {
                    'fold': fold_id,
                    'checkpoint': checkpoint_type,
                    'epoch': metadata.get('epoch', 0),
                    'threshold_f2_best': metadata.get('val_threshold_f2_best', 0.5),
                    'disp_MAE': metrics['displacement']['mae'],
                    'disp_RMSE': metrics['displacement']['rmse'],
                    'disp_R2': metrics['displacement']['r2'],
                    'event_PR_AUC': metrics['creep_burst_event']['pr_auc'],
                    'event_ROC_AUC': metrics['creep_burst_event']['roc_auc'],
                    'event_detection_rate': metrics['creep_burst_event']['detection_rate'],
                    'strict_FPR': metrics['creep_burst_event']['strict_fpr'],
                    'loose_FPR': metrics['creep_burst_event']['loose_fpr'],
                    'mean_lead_days': metrics['creep_burst_event']['mean_lead_time']
                }
                
                results.append(result)
                logger.info(f"Evaluated {checkpoint_type}_fold{fold_id}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {checkpoint_type}_fold{fold_id}: {e}")
                continue
        
        return results
    
    def _load_model_from_checkpoint(self, checkpoint):
        """从checkpoint加载模型"""
        from models.pi_phm import PIPHM
        
        # 重新创建模型（需要feature_index_map）
        # 这里需要从配置中获取feature_index_map
        # 简化处理：假设模型已经正确保存了state_dict
        model_state_dict = checkpoint['model_state_dict']
        
        # 创建新模型实例
        # 注意：这里需要正确的feature_index_map，可能需要从训练时保存
        # 作为简化，我们假设可以从checkpoint中恢复
        model = PIPHM.from_config(self.config, self._get_feature_index_map(), 
                                input_channels=len(self._get_static_features()))
        model.load_state_dict(model_state_dict)
        return model
    
    def _get_feature_index_map(self):
        """获取特征索引映射（简化实现）"""
        # 这里应该从训练配置中获取，但为了简化，返回空字典
        # 实际使用时需要正确实现
        return {}
    
    def _get_static_features(self):
        """获取静态特征（简化实现）"""
        return []
    
    def generate_detailed_evaluation_table(self, all_fold_results: List[List[Dict]]) -> pd.DataFrame:
        """
        生成详细的评估表
        
        Args:
            all_fold_results: 所有fold的评估结果
            
        Returns:
            pd.DataFrame: 详细评估表
        """
        # 展平所有结果
        flat_results = []
        for fold_results in all_fold_results:
            flat_results.extend(fold_results)
        
        # 创建DataFrame
        df = pd.DataFrame(flat_results)
        
        # 按fold和checkpoint排序
        df = df.sort_values(['fold', 'checkpoint']).reset_index(drop=True)
        
        return df
    
    def generate_summary_statistics(self, detailed_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成汇总统计表（mean ± std）
        
        Args:
            detailed_df: 详细评估表
            
        Returns:
            pd.DataFrame: 汇总统计表
        """
        summary_data = []
        checkpoint_types = ['best_disp', 'best_event', 'best_multi', 'last']
        
        for checkpoint_type in checkpoint_types:
            type_mask = detailed_df['checkpoint'] == checkpoint_type
            if type_mask.sum() == 0:
                continue
                
            type_data = detailed_df[type_mask]
            
            # 计算均值和标准差
            disp_mae_mean = type_data['disp_MAE'].mean()
            disp_mae_std = type_data['disp_MAE'].std()
            
            disp_rmse_mean = type_data['disp_RMSE'].mean()
            disp_rmse_std = type_data['disp_RMSE'].std()
            
            disp_r2_mean = type_data['disp_R2'].mean()
            disp_r2_std = type_data['disp_R2'].std()
            
            pr_auc_mean = type_data['event_PR_AUC'].mean()
            pr_auc_std = type_data['event_PR_AUC'].std()
            
            roc_auc_mean = type_data['event_ROC_AUC'].mean()
            roc_auc_std = type_data['event_ROC_AUC'].std()
            
            detection_rate_mean = type_data['event_detection_rate'].mean()
            detection_rate_std = type_data['event_detection_rate'].std()
            
            strict_fpr_mean = type_data['strict_FPR'].mean()
            strict_fpr_std = type_data['strict_FPR'].std()
            
            mean_lead_days_mean = type_data['mean_lead_days'].mean()
            mean_lead_days_std = type_data['mean_lead_days'].std()
            
            summary_row = {
                'checkpoint_type': checkpoint_type,
                'disp_MAE_mean±std': f"{disp_mae_mean:.3f}±{disp_mae_std:.3f}",
                'disp_RMSE_mean±std': f"{disp_rmse_mean:.3f}±{disp_rmse_std:.3f}",
                'disp_R2_mean±std': f"{disp_r2_mean:.3f}±{disp_r2_std:.3f}",
                'PR_AUC_mean±std': f"{pr_auc_mean:.3f}±{pr_auc_std:.3f}",
                'ROC_AUC_mean±std': f"{roc_auc_mean:.3f}±{roc_auc_std:.3f}",
                'detection_rate_mean±std': f"{detection_rate_mean:.3f}±{detection_rate_std:.3f}",
                'strict_FPR_mean±std': f"{strict_fpr_mean:.3f}±{strict_fpr_std:.3f}",
                'mean_lead_days_mean±std': f"{mean_lead_days_mean:.3f}±{mean_lead_days_std:.3f}"
            }
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def save_evaluation_results(self, detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
        """保存评估结果"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存详细评估表
        detailed_path = os.path.join(self.output_dir, "rolling_validation_detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed evaluation results saved to {detailed_path}")
        
        # 保存汇总统计表
        summary_path = os.path.join(self.output_dir, "rolling_validation_summary_statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics saved to {summary_path}")
        
        # 保存为Markdown格式便于查看
        detailed_md_path = os.path.join(self.output_dir, "rolling_validation_detailed_results.md")
        summary_md_path = os.path.join(self.output_dir, "rolling_validation_summary_statistics.md")
        
        with open(detailed_md_path, 'w') as f:
            f.write("# Rolling Validation Detailed Results\n\n")
            f.write(detailed_df.to_markdown(index=False))
        
        with open(summary_md_path, 'w') as f:
            f.write("# Rolling Validation Summary Statistics\n\n")
            f.write(summary_df.to_markdown(index=False))
        
        logger.info(f"Markdown results saved to {self.output_dir}")


def main():
    """主函数 - 执行rolling validation评估"""
    import torch
    from config import PI_PHM_Config
    
    # 加载配置
    config = PI_PHM_Config.from_yaml("config.yaml")
    
    # 创建评估器
    evaluator = RollingValidationEvaluator(config)
    
    # 这里需要加载每个fold的验证数据
    # 由于实现复杂，这里提供框架代码
    print("Rolling Validation Evaluator initialized")
    print("To run full evaluation, integrate with rolling_validation.py")


if __name__ == "__main__":
    main()