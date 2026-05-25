import os
import json
import torch
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Checkpoint管理器，负责保存和管理四类不同的最佳模型"""
    
    def __init__(self, save_dir: str, fold_id: Optional[int] = None):
        self.save_dir = save_dir
        self.fold_id = fold_id
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化最佳指标跟踪
        self.best_disp_mae = float('inf')
        self.best_event_prauc = 0.0
        self.best_multi_score = float('inf')
        
        # 记录每个checkpoint的元数据
        self.checkpoint_metadata = {
            'best_disp': None,
            'best_event': None,
            'best_multi': None,
            'last': None
        }
    
    def _get_checkpoint_filename(self, save_reason: str) -> str:
        """获取checkpoint文件名，支持fold_id"""
        if self.fold_id is not None:
            return f"{save_reason}_fold{self.fold_id}"
        else:
            return save_reason
    
    def should_save_best_disp(self, val_disp_mae_mm: float) -> bool:
        """判断是否应该保存best_disp checkpoint"""
        return val_disp_mae_mm < self.best_disp_mae
    
    def should_save_best_event(self, val_event_prauc: float) -> bool:
        """判断是否应该保存best_event checkpoint"""
        return val_event_prauc > self.best_event_prauc
    
    def should_save_best_multi(self, score_multi: float) -> bool:
        """判断是否应该保存best_multi checkpoint"""
        return score_multi < self.best_multi_score
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       phase_name: str,
                       metrics: Dict[str, Any],
                       save_reason: str):
        """保存checkpoint及其元数据"""
        filename_base = self._get_checkpoint_filename(save_reason)
        checkpoint_path = os.path.join(self.save_dir, f"{filename_base}.pth")
        metadata_path = os.path.join(self.save_dir, f"{filename_base}_metadata.json")
        
        # 保存模型checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'phase_name': phase_name,
            'fold_id': self.fold_id
        }
        torch.save(checkpoint, checkpoint_path)
        
        # 保存元数据
        metadata = {
            'fold_id': self.fold_id,
            'epoch': epoch,
            'phase_name': phase_name,
            'val_disp_mae_mm': metrics.get('val_disp_mae_mm', None),
            'val_event_pr_auc': metrics.get('val_event_prauc', None),  # P3要求的字段名
            'val_event_roc_auc': metrics.get('val_event_aucroc', None),  # P3要求的字段名
            'val_threshold_f2_best': metrics.get('val_threshold_f2_best', None),
            'val_threshold_strict': metrics.get('val_threshold_strict', None),
            'val_threshold_loose': metrics.get('val_threshold_loose', None),
            'threshold_source': metrics.get('threshold_source', 'validation'),  # 明确标记来源
            'val_f2_at_threshold_f2_best': metrics.get('val_f2_at_f2_best', None),
            'val_recall_at_threshold_f2_best': metrics.get('val_recall_at_f2_best', None),
            'val_fpr_at_threshold_f2_best': metrics.get('val_fpr_at_f2_best', None),
            'val_recall_at_threshold_strict': metrics.get('val_recall_at_strict', None),
            'val_fpr_at_threshold_strict': metrics.get('val_fpr_at_strict', None),
            'val_recall_at_threshold_loose': metrics.get('val_recall_at_loose', None),
            'val_fpr_at_threshold_loose': metrics.get('val_fpr_at_loose', None),
            'val_score_multi': metrics.get('val_score_multi', None),
            'validation_set_info': metrics.get('validation_set_info', None)  # 添加验证集统计信息
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.checkpoint_metadata[save_reason] = metadata
        logger.info(f"Saved {filename_base} checkpoint to {checkpoint_path}")
        
        # 更新最佳指标跟踪
        if save_reason == 'best_disp':
            self.best_disp_mae = metrics.get('val_disp_mae_mm', float('inf'))
        elif save_reason == 'best_event':
            self.best_event_prauc = metrics.get('val_event_prauc', 0.0)
        elif save_reason == 'best_multi':
            self.best_multi_score = metrics.get('val_score_multi', float('inf'))
    
    def save_last_checkpoint(self, 
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           epoch: int,
                           phase_name: str,
                           metrics: Dict[str, Any]):
        """保存最后一个epoch的checkpoint"""
        self.save_checkpoint(model, optimizer, epoch, phase_name, metrics, 'last')
    
    def get_all_checkpoints_info(self) -> Dict[str, Any]:
        """获取所有checkpoint的信息"""
        return self.checkpoint_metadata