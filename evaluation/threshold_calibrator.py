import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

class EventThresholdCalibrator:
    """事件检测阈值校准器 - 严格只使用验证集进行阈值校准"""
    
    def __init__(self, output_dir: Optional[str] = None, allow_default_threshold: bool = False):
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.allow_default_threshold = allow_default_threshold
        self.threshold_search_table = None
        self.threshold_f2_best = None
        self.threshold_strict = None
        self.threshold_loose = None
        self.is_fitted = False
        # 新增元数据字段
        self.val_pr_auc = None
        self.val_roc_auc = None
        self.val_recall_at_f2_best = None
        self.val_fpr_at_f2_best = None
        self.source_split = None
    
    def fit(self, y_true: np.ndarray, y_pred_prob: np.ndarray, data_split: str = "validation"):
        """
        在验证集上校准阈值
        
        Args:
            y_true: 验证集真实标签 (0/1)
            y_pred_prob: 验证集预测概率
            data_split: 数据分割名称，必须为"validation"以防止作弊
            
        Raises:
            RuntimeError: 如果data_split不是"validation"，防止使用测试集调阈值
        """
        # P2硬性要求：如果 calibrator.fit() 被喂了 test split，直接 raise RuntimeError
        if data_split != "validation":
            raise RuntimeError(f"Threshold calibration only allowed on validation set! "
                             f"Received data_split='{data_split}'. This is an anti-cheating measure.")
        
        # 验证输入数据
        if len(y_true) != len(y_pred_prob):
            raise ValueError("y_true and y_pred_prob must have the same length")
        
        # 计算AUC指标
        unique_labels = np.unique(y_true)
        if len(unique_labels) > 1:
            self.val_roc_auc = roc_auc_score(y_true, y_pred_prob)
            self.val_pr_auc = average_precision_score(y_true, y_pred_prob)
        else:
            self.val_roc_auc = 0.5  # 只有一个类别时AUC为0.5
            self.val_pr_auc = 0.0 if unique_labels[0] == 0 else 1.0
        
        self.source_split = data_split
        
        # 检查是否允许默认阈值
        if not self.allow_default_threshold:
            if len(unique_labels) <= 1:
                raise ValueError("Validation set must contain both positive and negative samples for threshold calibration. "
                               "Set allow_default_threshold=True to allow fallback to default thresholds.")
        
        if not np.array_equal(np.unique(y_true), [0, 1]):
            logger.warning("Validation set does not contain both positive and negative samples. "
                          "Threshold calibration may be unreliable.")
        
        # 生成阈值搜索表
        self.threshold_search_table = self._generate_threshold_search_table(y_true, y_pred_prob)
        
        # 计算三种阈值
        self.threshold_f2_best = self._find_f2_best_threshold()
        self.threshold_strict = self._find_strict_threshold()
        self.threshold_loose = self._find_loose_threshold()
        
        # 初始化指标变量
        self.val_f2_at_f2_best = None
        self.val_recall_at_f2_best = None
        self.val_fpr_at_f2_best = None
        self.val_recall_at_strict = None
        self.val_fpr_at_strict = None
        self.val_recall_at_loose = None
        self.val_fpr_at_loose = None
        
        # 计算各阈值下的指标
        if self.threshold_f2_best is not None:
            pred_binary_f2 = (y_pred_prob >= self.threshold_f2_best).astype(int)
            tp_f2 = np.sum((pred_binary_f2 == 1) & (y_true == 1))
            fp_f2 = np.sum((pred_binary_f2 == 1) & (y_true == 0))
            tn_f2 = np.sum((pred_binary_f2 == 0) & (y_true == 0))
            fn_f2 = np.sum((pred_binary_f2 == 0) & (y_true == 1))
            
            precision_f2 = tp_f2 / (tp_f2 + fp_f2) if (tp_f2 + fp_f2) > 0 else 0.0
            recall_f2 = tp_f2 / (tp_f2 + fn_f2) if (tp_f2 + fn_f2) > 0 else 0.0
            self.val_f2_at_f2_best = 5 * precision_f2 * recall_f2 / (4 * precision_f2 + recall_f2) if (4 * precision_f2 + recall_f2) > 0 else 0.0
            self.val_recall_at_f2_best = recall_f2
            self.val_fpr_at_f2_best = fp_f2 / (fp_f2 + tn_f2) if (fp_f2 + tn_f2) > 0 else 0.0
        
        if self.threshold_strict is not None:
            pred_binary_strict = (y_pred_prob >= self.threshold_strict).astype(int)
            tp_strict = np.sum((pred_binary_strict == 1) & (y_true == 1))
            fp_strict = np.sum((pred_binary_strict == 1) & (y_true == 0))
            tn_strict = np.sum((pred_binary_strict == 0) & (y_true == 0))
            fn_strict = np.sum((pred_binary_strict == 0) & (y_true == 1))
            
            self.val_recall_at_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
            self.val_fpr_at_strict = fp_strict / (fp_strict + tn_strict) if (fp_strict + tn_strict) > 0 else 0.0
        
        if self.threshold_loose is not None:
            pred_binary_loose = (y_pred_prob >= self.threshold_loose).astype(int)
            tp_loose = np.sum((pred_binary_loose == 1) & (y_true == 1))
            fp_loose = np.sum((pred_binary_loose == 1) & (y_true == 0))
            tn_loose = np.sum((pred_binary_loose == 0) & (y_true == 0))
            fn_loose = np.sum((pred_binary_loose == 0) & (y_true == 1))
            
            self.val_recall_at_loose = tp_loose / (tp_loose + fn_loose) if (tp_loose + fn_loose) > 0 else 0.0
            self.val_fpr_at_loose = fp_loose / (fp_loose + tn_loose) if (fp_loose + tn_loose) > 0 else 0.0
        
        self.is_fitted = True
        
        # 保存阈值搜索表
        if self.output_dir:
            search_table_path = os.path.join(self.output_dir, "threshold_search_table.csv")
            self.threshold_search_table.to_csv(search_table_path, index=False)
            logger.info(f"Threshold search table saved to {search_table_path}")
            
            # 保存阈值元数据
            metadata_path = os.path.join(self.output_dir, "threshold_metadata.json")
            metadata = {
                'threshold_f2_best': self.threshold_f2_best,
                'threshold_strict': self.threshold_strict,
                'threshold_loose': self.threshold_loose,
                'source_split': self.source_split,
                'val_pr_auc': self.val_pr_auc,
                'val_roc_auc': self.val_roc_auc,
                'val_f2_at_threshold_f2_best': self.val_f2_at_f2_best,
                'val_recall_at_threshold_f2_best': self.val_recall_at_f2_best,
                'val_fpr_at_threshold_f2_best': self.val_fpr_at_f2_best,
                'val_recall_at_threshold_strict': self.val_recall_at_strict,
                'val_fpr_at_threshold_strict': self.val_fpr_at_strict,
                'val_recall_at_threshold_loose': self.val_recall_at_loose,
                'val_fpr_at_threshold_loose': self.val_fpr_at_loose
            }
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            logger.info(f"Threshold metadata saved to {metadata_path}")
        
        return self
    
    def _generate_threshold_search_table(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> pd.DataFrame:
        """生成阈值搜索表"""
        thresholds = np.arange(0.05, 0.96, 0.01)  # [0.05, 0.95] with step 0.01
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            # 计算混淆矩阵
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2,
                'fpr': fpr,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        return pd.DataFrame(results)
    
    def _find_f2_best_threshold(self) -> float:
        """找到F2分数最高的阈值"""
        if self.threshold_search_table is None:
            raise ValueError("Must call fit() before getting thresholds")
        
        # 找到F2分数最高的阈值
        best_f2_idx = self.threshold_search_table['f2'].idxmax()
        best_f2_value = self.threshold_search_table.loc[best_f2_idx, 'f2']
        
        # 如果有多个阈值具有相同的F2分数，选择FPR更低的
        candidates = self.threshold_search_table[
            self.threshold_search_table['f2'] == best_f2_value
        ].copy()
        
        if len(candidates) > 1:
            # 按FPR升序，如果FPR相同则按阈值降序（更保守）
            candidates = candidates.sort_values(['fpr', 'threshold'], ascending=[True, False])
            best_f2_idx = candidates.index[0]
        
        return self.threshold_search_table.loc[best_f2_idx, 'threshold']
    
    def _find_strict_threshold(self) -> float:
        """找到满足FPR <= 0.10且recall最大的阈值"""
        if self.threshold_search_table is None:
            raise ValueError("Must call fit() before getting thresholds")
        
        # 筛选FPR <= 0.10的阈值
        strict_candidates = self.threshold_search_table[
            self.threshold_search_table['fpr'] <= 0.10
        ]
        
        if len(strict_candidates) > 0:
            # 选择recall最大的
            best_recall_idx = strict_candidates['recall'].idxmax()
            return strict_candidates.loc[best_recall_idx, 'threshold']
        else:
            # 如果没有满足条件的，选择FPR最接近0.10且recall最大的
            # 计算FPR与0.10的距离
            distances = np.abs(self.threshold_search_table['fpr'] - 0.10)
            min_distance = distances.min()
            closest_candidates = self.threshold_search_table[distances == min_distance]
            
            # 在距离最近的候选者中选择recall最大的
            best_recall_idx = closest_candidates['recall'].idxmax()
            return closest_candidates.loc[best_recall_idx, 'threshold']
    
    def _find_loose_threshold(self) -> float:
        """找到满足FPR <= 0.20且recall最大的阈值"""
        if self.threshold_search_table is None:
            raise ValueError("Must call fit() before getting thresholds")
        
        # 筛选FPR <= 0.20的阈值
        loose_candidates = self.threshold_search_table[
            self.threshold_search_table['fpr'] <= 0.20
        ]
        
        if len(loose_candidates) > 0:
            # 选择recall最大的
            best_recall_idx = loose_candidates['recall'].idxmax()
            return loose_candidates.loc[best_recall_idx, 'threshold']
        else:
            # 如果没有满足条件的，选择FPR最接近0.20且recall最大的
            # 计算FPR与0.20的距离
            distances = np.abs(self.threshold_search_table['fpr'] - 0.20)
            min_distance = distances.min()
            closest_candidates = self.threshold_search_table[distances == min_distance]
            
            # 在距离最近的候选者中选择recall最大的
            best_recall_idx = closest_candidates['recall'].idxmax()
            return closest_candidates.loc[best_recall_idx, 'threshold']
    
    def get_thresholds(self) -> Dict[str, float]:
        """获取校准后的阈值"""
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting thresholds")
        
        return {
            'threshold_f2_best': self.threshold_f2_best,
            'threshold_strict': self.threshold_strict,
            'threshold_loose': self.threshold_loose
        }
    
    def get_metadata(self) -> Dict:
        """获取校准后的完整元数据"""
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting metadata")
        
        return {
            'threshold_f2_best': self.threshold_f2_best,
            'threshold_strict': self.threshold_strict,
            'threshold_loose': self.threshold_loose,
            'source_split': self.source_split,
            'val_pr_auc': self.val_pr_auc,
            'val_roc_auc': self.val_roc_auc,
            'val_f2_at_threshold_f2_best': self.val_f2_at_f2_best,
            'val_recall_at_threshold_f2_best': self.val_recall_at_f2_best,
            'val_fpr_at_threshold_f2_best': self.val_fpr_at_f2_best,
            'val_recall_at_threshold_strict': self.val_recall_at_strict,
            'val_fpr_at_threshold_strict': self.val_fpr_at_strict,
            'val_recall_at_threshold_loose': self.val_recall_at_loose,
            'val_fpr_at_threshold_loose': self.val_fpr_at_loose
        }
    
    def get_threshold_search_table(self) -> pd.DataFrame:
        """获取阈值搜索表"""
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting threshold search table")
        
        return self.threshold_search_table.copy()