import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from collections import Counter
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from evaluation.threshold_calibrator import EventThresholdCalibrator

# 导入FocalLoss和EarlyStopping
from .losses import FocalLoss
from utils.utils import EarlyStopping
from .checkpoint_manager import CheckpointManager

# 使用Python内置logging
logger = logging.getLogger(__name__)

# 假设utils中包含EarlyStopping类
try:
    from utils import EarlyStopping
except ImportError:
    # 如果没有utils模块，创建一个简单的EarlyStopping
    class EarlyStopping:
        def __init__(self, patience=20, min_delta=1e-5, monitor="val_combined_score"):
            self.patience = patience
            self.min_delta = min_delta
            self.monitor = monitor
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            
        def __call__(self, score):
            if self.best_score is None:
                self.best_score = score
            elif score >= self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

class CurriculumScheduler:
    """课程学习调度器 - 任务6更新版本"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        self.config = config
        # 如果配置中提供了curriculum_stages和stage_weights，则使用配置的版本
        if (hasattr(config.training, 'curriculum_stages') and 
            hasattr(config.training, 'stage_weights') and
            config.training.curriculum_stages and 
            config.training.stage_weights):
            # 使用配置的阶段边界和权重
            self.curriculum_stages = config.training.curriculum_stages
            self.stage_weights = config.training.stage_weights
            self.use_custom = True
            print(f"Using custom curriculum phases from config: stages={self.curriculum_stages}")
        else:
            self.use_custom = False
            print("Using task6 curriculum phases")
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        根据当前epoch返回损失权重
        
        P4.2 保守课程学习策略：
        Phase 1 (epoch 0-19): 只训练位移主任务
          L = L_disp
          event loss = 0.0, risk loss = 0.0, physics losses = 0.0
        
        Phase 2 (epoch 20-59): 小权重接入事件检测
          L = L_disp + 0.10 * L_event
          risk loss = 0.0, physics losses = 0.0
        
        Phase 3 (epoch 60-99): 逐渐引入辅助任务
          L = L_disp + 0.20 * L_event + 0.10 * L_risk
          physics losses = 0.001 ~ 0.005（低权重）
        
        Phase 4 (epoch 100+): 精调
          L = L_disp + 0.20 * L_event + 0.10 * L_risk + physics small weights
          学习率下降10倍
        """
        if self.use_custom:
            # 根据curriculum_stages确定当前阶段
            current_phase = 0
            for i, stage_end in enumerate(self.curriculum_stages):
                if epoch < stage_end:
                    current_phase = i
                    break
            else:
                current_phase = len(self.curriculum_stages) - 1
            
            # 获取当前阶段的权重配置
            if current_phase < len(self.stage_weights):
                phase_config = self.stage_weights[current_phase]
            else:
                # 如果阶段超出配置，使用最后一个阶段的配置
                phase_config = self.stage_weights[-1]
            
            # 转换为标准格式
            return {
                'alpha_risk': phase_config.get('alpha_risk', 0.0),
                'lambda_creep': phase_config.get('lambda_creep', 0.0),
                'lambda_stress': phase_config.get('lambda_stress', 0.0),
                'lambda_seismic': phase_config.get('lambda_seismic', 0.0),
                'lambda_event': phase_config.get('lambda_event', 0.0),
                'lambda_causal': phase_config.get('lambda_causal', 0.0)
            }
        else:
            # P4.2 保守策略的硬编码版本（备用）
            if epoch < 20:  # Phase 1
                return {
                    'alpha_risk': 0.0,      # 不加风险分类
                    'lambda_creep': 0.0,    # 不加物理约束
                    'lambda_stress': 0.0,
                    'lambda_seismic': 0.0,
                    'lambda_event': 0.0,    # 事件检测权重为0
                    'lambda_causal': 0.0    # 不加因果约束
                }
            elif epoch < 60:  # Phase 2
                return {
                    'alpha_risk': 1.0,      # 开启风险分类
                    'lambda_creep': 0.0,    # 不加物理约束
                    'lambda_stress': 0.0,
                    'lambda_seismic': 0.0,
                    'lambda_event': 0.4,    # 恢复为0.4：小权重事件检测
                    'lambda_causal': 0.0    # 不加因果约束
                }
            elif epoch < 100:  # Phase 3
                return {
                    'alpha_risk': 1.0,      # 开启风险分类
                    'lambda_creep': 0.001,  # 低权重蠕变约束
                    'lambda_stress': 0.001, # 低权重应力约束
                    'lambda_seismic': 0.001, # 低权重微震约束
                    'lambda_event': 0.6,    # 恢复为0.6：中等权重事件检测
                    'lambda_causal': 0.0    # 因果约束在Phase 4才加入
                }
            else:  # Phase 4
                return {
                    'alpha_risk': 1.0,      # 开启风险分类
                    'lambda_creep': 0.01,   # 蠕变约束
                    'lambda_stress': 0.005, # 应力约束
                    'lambda_seismic': 0.005, # 微震约束
                    'lambda_event': 1.0,    # 正常权重事件检测
                    'lambda_causal': 0.01   # 因果约束
                }
            
    def get_phase_name(self, epoch: int) -> str:
        """获取当前阶段名称"""
        if self.use_custom:
            current_phase = 0
            for i, stage_end in enumerate(self.curriculum_stages):
                if epoch < stage_end:
                    current_phase = i
                    break
            else:
                current_phase = len(self.curriculum_stages) - 1
            
            phase_names = [
                "Phase 1: 仅位移主任务",
                "Phase 2: 小权重事件检测", 
                "Phase 3: 引入辅助任务",
                "Phase 4: 全权重精调"
            ]
            if current_phase < len(phase_names):
                return phase_names[current_phase]
            else:
                return f"Phase {current_phase + 1}: 全权重精调"
        else:
            if epoch < 20:
                return "Phase 1: 仅位移主任务"
            elif epoch < 60:
                return "Phase 2: 小权重事件检测"
            elif epoch < 100:
                return "Phase 3: 引入辅助任务"
            else:
                return "Phase 4: 全权重精调"
    
    def get_current_phase(self, epoch: int) -> int:
        """获取当前阶段编号"""
        if self.use_custom:
            current_phase = 0
            for i, stage_end in enumerate(self.curriculum_stages):
                if epoch < stage_end:
                    current_phase = i
                    break
            else:
                current_phase = len(self.curriculum_stages) - 1
            return current_phase + 1
        else:
            if epoch < 20:
                return 1
            elif epoch < 60:
                return 2
            elif epoch < 100:
                return 3
            else:
                return 4


class PI_PHM_Trainer:
    """PI-PHM训练器"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 config: 'PI_PHM_Config', device: torch.device, feature_index_map: Dict[str, Union[int, List[int]]],
                 normalizer: Optional[object] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.normalizer = normalizer  # 添加normalizer属性
        
        # 初始化模型到设备
        self.model.to(self.device)
        
        # 初始化checkpoint manager
        checkpoint_dir = 'outputs/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 初始化阈值校准器
        calibrator_output_dir = os.path.join('outputs', 'calibration')
        if hasattr(self, 'fold_id'):
            calibrator_output_dir = os.path.join('outputs', 'calibration', f'fold_{getattr(self, "fold_id", "main")}')
        self.threshold_calibrator = EventThresholdCalibrator(
            output_dir=calibrator_output_dir,
            allow_default_threshold=False  # 严格模式，不允许fallback
        )
        
        # P4.3: 事件头微调机制相关属性
        self.event_finetune_enabled = False
        self.event_finetune_triggered = False
        self.event_pr_auc_history = []
        self.event_finetune_patience = 15  # 连续15个epoch无提升则触发
        
        # 初始化优化器
        # 确保参数类型正确，提供默认值
        lr_value = float(getattr(config.training, 'lr', 1e-4))
        weight_decay_value = float(getattr(config.training, 'weight_decay', 1e-5))
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_value,
            weight_decay=weight_decay_value
        )
        
        # 学习率调度器 - 简化处理
        # 如果配置中没有scheduler参数，使用简单的StepLR
        if hasattr(config.training, 'scheduler_T0'):
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.training.scheduler_T0,
                T_mult=config.training.scheduler_Tmult,
                eta_min=config.training.scheduler_eta_min
            )
        else:
            # 使用简单的StepLR作为默认
            self.cosine_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        
        # Warmup配置（注意：原实现创建了scheduler但未step，导致LR长期固定在base_lr/5）
        self.base_lr = lr_value
        self.warmup_epochs = 5
        # 显式设置初始LR为warmup起点，避免依赖LambdaLR的隐式行为
        initial_lr = self.base_lr / self.warmup_epochs
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
        self.warmup_scheduler = None
        
        # 梯度裁剪参数
        self.max_norm = config.training.grad_clip
        
        # 早停：使用phase-invariant monitor metric，避免跨phase的score定义变化影响early stopping
        self.monitor_metric = getattr(getattr(config, 'early_stopping', {}), 'monitor', 'val_score_multi')
        if self.monitor_metric == 'val_combined_score':
            self.monitor_metric = 'val_score_multi'
        self.early_stopping = EarlyStopping(
            patience=config.training.phase1_patience,
            min_delta=1e-5,
            monitor=self.monitor_metric
        )
        self._phase2_patience_updated = False
        
        # TensorBoard
        log_dir = os.path.join("outputs", "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir)
        
        # 损失函数
        from .losses import PIPHMLoss
        # 从配置中获取focal_alpha参数
        focal_alpha = getattr(self.config.training, 'focal_alpha', None)
        print(f"Config focal_alpha: {focal_alpha}")
        self.criterion = PIPHMLoss(feature_index_map, focal_alpha=focal_alpha)
        
        # 课程学习调度器
        self.curriculum_scheduler = CurriculumScheduler(config)
        
        # 最佳监控分数（越小越好）
        self.best_combined_score = float('inf')
        
        # MAE基线（用于计算combined score）
        self.mae_baseline = None
        
        # 检查点目录
        self.checkpoint_dir = os.path.join("outputs", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 当前epoch
        self.current_epoch = 0
        
        # Persistence增量基线MAE（在第一次验证时计算）
        self.persistence_mae_baseline = None
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
        
    def _compute_persistence_increment_baseline(self, true_disp: torch.Tensor) -> float:
        """计算Persistence增量基线MAE"""
        # 转换为numpy进行计算
        true_disp_np = true_disp.numpy()
        
        # 确保是2D数组 (batch_size, forecast_days)
        if len(true_disp_np.shape) == 1:
            true_disp_np = true_disp_np.reshape(-1, 1)
            
        if len(true_disp_np) < 2:
            return float('inf')
        
        # 计算每日增量（只考虑第一天的预测）
        daily_increments = np.diff(true_disp_np[:, 0])  
        
        # Persistence增量预测：用最近7天平均日增量预测未来
        if len(daily_increments) >= 7:
            avg_increment = np.mean(daily_increments[-7:])
        else:
            avg_increment = np.mean(daily_increments) if len(daily_increments) > 0 else 0.0
        
        # 预测的增量序列
        pred_increments = np.full_like(true_disp_np[1:, 0], avg_increment)
        true_increments = daily_increments
        
        if len(true_increments) == 0:
            return float('inf')
            
        mae = np.mean(np.abs(pred_increments - true_increments))
        return mae if not np.isnan(mae) else float('inf')
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch，返回平均总损失"""
        self.model.train()
        
        # 存储详细损失用于TensorBoard记录
        self.epoch_loss_components = {
            'loss_disp': 0.0,
            'loss_aux': 0.0,
            'loss_risk': 0.0,
            'loss_event': 0.0,
            'loss_creep': 0.0,
            'loss_stress': 0.0,
            'loss_seismic': 0.0,
            'loss_causal': 0.0,
            # weighted contributions to total loss (more interpretable across curriculum phases)
            'wloss_disp': 0.0,
            'wloss_aux': 0.0,
            'wloss_risk': 0.0,
            'wloss_event': 0.0,
            'wloss_creep': 0.0,
            'wloss_stress': 0.0,
            'wloss_seismic': 0.0,
            'wloss_causal': 0.0,
        }
        
        total_loss = 0.0
        num_batches = 0
        
        # P4.1: 训练信号审计 - 前3个epoch的前5个batch
        audit_batches = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 将batch数据移到设备
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(self.device)
                else:
                    batch_device[key] = value
                    
            # 前向传播
            try:
                model_inputs = {
                    'x_dynamic': batch_device['x_dynamic'],
                    'x_static': batch_device['x_static']
                }
                if 'mask' in batch_device:
                    model_inputs['mask'] = batch_device['mask']
                
                model_outputs = self.model(**model_inputs)
                
                # 获取当前epoch的损失权重
                loss_weights = self.curriculum_scheduler.get_loss_weights(epoch)
                
                # 计算损失 - PIPHMLoss返回(total_loss, loss_dict)
                loss_result = self.criterion(model_outputs, batch_device, loss_weights)
                if isinstance(loss_result, tuple):
                    total_loss_batch, loss_dict_batch = loss_result
                else:
                    total_loss_batch = loss_result
                    loss_dict_batch = {}
                
                # 检查NaN
                if torch.isnan(total_loss_batch):
                    print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                
                # 梯度裁剪
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                
                self.optimizer.step()
                
                # 累积损失（兼容PIPHMLoss当前返回的key命名）
                total_loss += total_loss_batch.item()
                raw_disp = loss_dict_batch.get('disp_loss', loss_dict_batch.get('L_disp', 0.0))
                raw_aux = loss_dict_batch.get('aux_loss', 0.0)
                raw_risk = loss_dict_batch.get('risk_loss', loss_dict_batch.get('L_risk', 0.0))
                raw_event = loss_dict_batch.get('event_loss', loss_dict_batch.get('L_event', 0.0))
                raw_creep = loss_dict_batch.get('creep_constr', loss_dict_batch.get('L_creep', 0.0))
                raw_stress = loss_dict_batch.get('stress_constr', loss_dict_batch.get('L_stress', 0.0))
                raw_seismic = loss_dict_batch.get('seismic_constr', loss_dict_batch.get('L_seismic', 0.0))
                raw_causal = loss_dict_batch.get('causal_constr', 0.0)

                self.epoch_loss_components['loss_disp'] += raw_disp
                self.epoch_loss_components['loss_aux'] += raw_aux
                self.epoch_loss_components['loss_risk'] += raw_risk
                self.epoch_loss_components['loss_event'] += raw_event
                self.epoch_loss_components['loss_creep'] += raw_creep
                self.epoch_loss_components['loss_stress'] += raw_stress
                self.epoch_loss_components['loss_seismic'] += raw_seismic
                self.epoch_loss_components['loss_causal'] += raw_causal

                self.epoch_loss_components['wloss_disp'] += raw_disp
                self.epoch_loss_components['wloss_aux'] += raw_aux
                self.epoch_loss_components['wloss_risk'] += loss_weights.get('alpha_risk', 0.0) * raw_risk
                self.epoch_loss_components['wloss_event'] += loss_weights.get('lambda_event', 0.0) * raw_event
                self.epoch_loss_components['wloss_creep'] += loss_weights.get('lambda_creep', 0.0) * raw_creep
                self.epoch_loss_components['wloss_stress'] += loss_weights.get('lambda_stress', 0.0) * raw_stress
                self.epoch_loss_components['wloss_seismic'] += loss_weights.get('lambda_seismic', 0.0) * raw_seismic
                self.epoch_loss_components['wloss_causal'] += loss_weights.get('lambda_causal', 0.0) * raw_causal
                num_batches += 1
                
                # P4.1: 记录审计信息（前3个epoch，前5个batch）
                if epoch < 3 and batch_idx < 5:
                    audit_info = {
                        'batch_idx': batch_idx,
                        'loss_disp': loss_dict_batch.get('disp_loss', loss_dict_batch.get('L_disp', 0.0)),
                        'loss_event': loss_dict_batch.get('event_loss', loss_dict_batch.get('L_event', 0.0)),
                        'loss_risk': loss_dict_batch.get('risk_loss', loss_dict_batch.get('L_risk', 0.0)),
                        'total_loss': total_loss_batch.item()
                    }
                    
                    # 记录事件预测概率统计
                    if 'pred_event_logits' in model_outputs:
                        pred_event_probs = torch.sigmoid(model_outputs['pred_event_logits']).detach()
                        audit_info['pred_event_prob_mean'] = pred_event_probs.mean().item()
                        audit_info['pred_event_prob_std'] = pred_event_probs.std().item()
                        
                        # 事件正样本率
                        if 'y_event' in batch_device:
                            true_event_rate = batch_device['y_event'].float().mean().item()
                            audit_info['event_positive_rate_in_batch'] = true_event_rate
                            
                            # 预测正样本率 @ 0.5
                            pred_positive_rate_05 = (pred_event_probs >= 0.5).float().mean().item()
                            audit_info['event_pred_positive_rate_05'] = pred_positive_rate_05
                            
                            # 预测正样本率 @ threshold_f2_best（如果可用）
                            if hasattr(self, 'threshold_calibrator') and self.threshold_calibrator.is_fitted:
                                calibrated_thresholds = self.threshold_calibrator.get_thresholds()
                                threshold_f2_best = calibrated_thresholds['threshold_f2_best']
                                pred_positive_rate_f2 = (pred_event_probs >= threshold_f2_best).float().mean().item()
                                audit_info['event_pred_positive_rate_f2'] = pred_positive_rate_f2
                            else:
                                audit_info['event_pred_positive_rate_f2'] = None
                    
                    # 记录梯度范数（如果需要）
                    if epoch == 0 and batch_idx == 0:  # 只在第一个batch记录梯度范数
                        backbone_grad_norm = 0.0
                        event_head_grad_norm = 0.0
                        disp_head_grad_norm = 0.0
                        
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if 'backbone' in name or ('patchtst' in name) or ('gru' in name):
                                    backbone_grad_norm += grad_norm ** 2
                                elif 'event_head' in name:
                                    event_head_grad_norm += grad_norm ** 2
                                elif 'disp_head' in name:
                                    disp_head_grad_norm += grad_norm ** 2
                        
                        audit_info['backbone_grad_norm'] = backbone_grad_norm ** 0.5
                        audit_info['event_head_grad_norm'] = event_head_grad_norm ** 0.5
                        audit_info['disp_head_grad_norm'] = disp_head_grad_norm ** 0.5
                    
                    audit_batches.append(audit_info)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # P4.1: 打印审计日志
        if epoch < 3 and audit_batches:
            print(f"\n{'='*60}")
            print(f"TRAINING SIGNAL AUDIT - Epoch {epoch}")
            print(f"{'='*60}")
            for audit_info in audit_batches:
                print(f"Batch {audit_info['batch_idx']}:")
                print(f"  Loss - Disp: {audit_info['loss_disp']:.4f}, Event: {audit_info['loss_event']:.4f}, Risk: {audit_info['loss_risk']:.4f}, Total: {audit_info['total_loss']:.4f}")
                if 'pred_event_prob_mean' in audit_info:
                    print(f"  Pred Event Prob - Mean: {audit_info['pred_event_prob_mean']:.4f}, Std: {audit_info['pred_event_prob_std']:.4f}")
                    if 'event_positive_rate_in_batch' in audit_info:
                        print(f"  Event Rate - True: {audit_info['event_positive_rate_in_batch']:.4f}, Pred@0.5: {audit_info['event_pred_positive_rate_05']:.4f}")
                        if audit_info['event_pred_positive_rate_f2'] is not None:
                            print(f"  Pred@F2-best: {audit_info['event_pred_positive_rate_f2']:.4f}")
                if 'backbone_grad_norm' in audit_info:
                    print(f"  Grad Norm - Backbone: {audit_info['backbone_grad_norm']:.4f}, Event Head: {audit_info['event_head_grad_norm']:.4f}, Disp Head: {audit_info['disp_head_grad_norm']:.4f}")
            print(f"{'='*60}\n")
        
        # 计算平均损失
        if num_batches == 0:
            return float('inf')
            
        avg_total_loss = total_loss / num_batches
        for key in self.epoch_loss_components:
            self.epoch_loss_components[key] /= num_batches
            
        return avg_total_loss
        
    def finetune_classifier(self, n_epochs=5):
        """修复B：专项训练分类器"""
        logger.info(f"开始修复B：专项训练分类器，{n_epochs} epochs")
        
        # 冻结除risk_head以外的所有参数
        for name, param in self.model.named_parameters():
            if 'risk_head' not in name:
                param.requires_grad = False
        
        # 创建只包含YELLOW/RED样本的过滤数据加载器
        yellow_red_loader = self._create_filtered_loader(
            self.train_loader.dataset,
            filter_labels=[2, 3],  # YELLOW=2, RED=3
            batch_size=16
        )
        
        # 20倍放大的FocalLoss
        strong_focal = FocalLoss(
            gamma=3.0,
            alpha=[0.0, 0.0, 0.5, 0.5]  # 只关注YELLOW和RED，使用list而不是tensor
        ).to(self.device)
        
        # 专用优化器
        optimizer_cls = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )
        
        # 专项训练
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            batch_count = 0
            
            for batch in yellow_red_loader:
                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(self.device)
                    else:
                        batch_device[key] = value
                
                outputs = self.model(
                    batch_device['x_dynamic'],
                    batch_device['x_static'],
                    batch_device.get('mask', None)
                )
                
                loss = strong_focal(outputs['pred_risk_logits'], batch_device['y_risk'])
                
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"修复B epoch {epoch+1}/{n_epochs}, loss: {avg_loss:.4f}")
        
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("修复B完成，所有参数已解冻")
    
    def _create_filtered_loader(self, dataset, filter_labels, batch_size):
        """创建过滤后的数据加载器"""
        # 找到符合条件的样本索引
        filtered_indices = []
        for i in range(len(dataset)):
            # 获取样本的真实标签
            sample = dataset[i]
            y_risk = sample['y_risk']
            if y_risk in filter_labels:
                filtered_indices.append(i)
        
        if len(filtered_indices) == 0:
            logger.warning(f"警告：没有找到标签为{filter_labels}的样本")
            # 返回原始数据加载器作为后备
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
        
        # 创建子集
        from torch.utils.data import Subset
        filtered_dataset = Subset(dataset, filtered_indices)
        
        return DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch（任务6更新：支持事件检测评估）"""
        self.model.eval()
        all_pred_disp = []
        all_true_disp = []
        all_pred_risk = []
        all_true_risk = []
        all_pred_event_logits = []  # 新增事件检测logits
        all_true_event = []         # 新增事件标签
        
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    batch_device = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch_device[key] = value.to(self.device)
                        else:
                            batch_device[key] = value
                            
                    model_outputs = self.model(
                        batch_device['x_dynamic'],
                        batch_device['x_static'],
                        batch_device.get('mask', None)
                    )
                    
                    # 收集预测和真实值（仅当模型输出包含相应key时）
                    if 'pred_disp' in model_outputs:
                        all_pred_disp.append(model_outputs['pred_disp'].cpu())
                        all_true_disp.append(batch_device['y_disp_main'].cpu())
                    
                    # 收集风险分类数据（仅当模型输出包含pred_risk_logits时）
                    if 'pred_risk_logits' in model_outputs:
                        all_pred_risk.append(model_outputs['pred_risk_logits'].cpu())
                        all_true_risk.append(batch_device['y_risk'].cpu())
                    
                    # 收集事件检测相关数据
                    if 'pred_event_logits' in model_outputs and 'y_event' in batch_device:
                        all_pred_event_logits.append(model_outputs['pred_event_logits'].cpu())
                        all_true_event.append(batch_device['y_event'].cpu())
                    
            # 合并所有批次（仅当列表非空时）
            if len(all_pred_disp) > 0 and len(all_true_disp) > 0:
                pred_disp = torch.cat(all_pred_disp, dim=0)
                true_disp = torch.cat(all_true_disp, dim=0)
                
                # 收集事件检测相关数据
                print(f"DEBUG: all_pred_event_logits length: {len(all_pred_event_logits)}")
                print(f"DEBUG: all_true_event length: {len(all_true_event)}")
                
                if len(all_pred_event_logits) > 0 and len(all_true_event) > 0:
                    pred_event_logits = torch.cat(all_pred_event_logits, dim=0).numpy().flatten()
                    true_event_labels = torch.cat(all_true_event, dim=0).numpy().flatten()
                    
                    # 检查标签唯一值
                    unique_labels = np.unique(true_event_labels)
                    print(f"DEBUG: unique_labels: {unique_labels}")
                    print(f"DEBUG: pred_event_logits shape: {pred_event_logits.shape}")
                    print(f"DEBUG: true_event_labels shape: {true_event_labels.shape}")
                
                # 反归一化位移预测和真实值（使用增量反归一化）
                if self.normalizer is not None:
                    target_col = self.config.data.target_col
                    pred_disp_denorm = torch.FloatTensor([
                        self.normalizer.inverse_transform_increment(
                            pred_disp[i:i+1].numpy(), target_col
                        ).flatten() for i in range(pred_disp.shape[0])
                    ])
                    true_disp_denorm = torch.FloatTensor([
                        self.normalizer.inverse_transform_increment(
                            true_disp[i:i+1].numpy(), target_col
                        ).flatten() for i in range(true_disp.shape[0])
                    ])
                else:
                    # 如果没有normalizer，直接使用原始值
                    pred_disp_denorm = pred_disp
                    true_disp_denorm = true_disp
                
                # 计算位移指标（在物理空间）- 仅当位移预测存在时
                mae = torch.mean(torch.abs(pred_disp_denorm - true_disp_denorm)).item()
                rmse = torch.sqrt(torch.mean((pred_disp_denorm - true_disp_denorm) ** 2)).item()
                r2 = 1 - torch.sum((pred_disp_denorm - true_disp_denorm) ** 2) / torch.sum((true_disp_denorm - torch.mean(true_disp_denorm)) ** 2)
                r2 = r2.item()
                
                # 确保所有指标都是Python原生类型
                mae = float(mae)
                rmse = float(rmse)
                r2 = float(r2)
                
                # 计算持久性基线（仅当位移数据存在时）
                self.persistence_mae_baseline = self._compute_persistence_increment_baseline(true_disp_denorm)
            else:
                mae = float('nan')
                rmse = float('nan')
                r2 = float('nan')
                self.persistence_mae_baseline = float('nan')
            
            # 只有当风险预测存在时才合并风险相关数据
            if len(all_pred_risk) > 0 and len(all_true_risk) > 0:
                pred_risk = torch.cat(all_pred_risk, dim=0)
                true_risk = torch.cat(all_true_risk, dim=0)
            else:
                pred_risk = None
                true_risk = None
            
            # 初始化风险指标默认值
            accuracy = 0.0
            yellow_recall = 0.0
            red_recall = 0.0
            f1_weighted = 0.0
            f1_macro = 0.0
            
            # 只有当风险预测存在时才计算风险分类指标
            if pred_risk is not None and true_risk is not None:
                pred_risk_labels = torch.argmax(pred_risk, dim=1)
                accuracy = torch.mean((pred_risk_labels == true_risk).float()).item()
                
                # 计算各类别的recall和F1分数
                from sklearn.metrics import precision_recall_fscore_support, f1_score
                _, recall_per_class, _, _ = precision_recall_fscore_support(
                    true_risk.numpy(), pred_risk_labels.numpy(), 
                    average=None, labels=[0, 1, 2, 3], zero_division=0)
                yellow_recall = recall_per_class[2] if len(recall_per_class) > 2 else 0.0
                red_recall = recall_per_class[3] if len(recall_per_class) > 3 else 0.0
                
                f1_weighted = f1_score(true_risk.numpy(), pred_risk_labels.numpy(), average='weighted', zero_division=0)
                f1_macro = f1_score(true_risk.numpy(), pred_risk_labels.numpy(), average='macro', zero_division=0)
            
            # 转换为Python原生类型
            accuracy = float(accuracy)
            yellow_recall = float(yellow_recall)
            red_recall = float(red_recall)
            f1_weighted = float(f1_weighted)
            f1_macro = float(f1_macro)
            
            # 计算Persistence增量基线MAE（第一次验证时）
            if self.persistence_mae_baseline is None:
                self.persistence_mae_baseline = self._compute_persistence_increment_baseline(true_disp_denorm)
                if self.persistence_mae_baseline == float('inf'):
                    self.persistence_mae_baseline = mae  # 如果无法计算，使用当前MAE作为基线
            
            # 计算事件检测指标（如果存在事件数据）
            event_auc = 0.0
            event_prauc = 0.0
            calibrated_thresholds = None  # 初始化为None
            threshold_calibration_success = False
            validation_set_info = None  # 新增：验证集统计信息
            
            if len(all_pred_event_logits) > 0 and len(all_true_event) > 0:
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    pred_event_probs = torch.sigmoid(torch.cat(all_pred_event_logits, dim=0)).numpy().flatten()
                    true_event_labels = torch.cat(all_true_event, dim=0).numpy().flatten()
                    
                    # 计算验证集统计信息
                    total_samples = len(true_event_labels)
                    pos_samples = int(np.sum(true_event_labels))
                    neg_samples = total_samples - pos_samples
                    pos_ratio = f"{(pos_samples / total_samples) * 100:.1f}%"
                    neg_ratio = f"{(neg_samples / total_samples) * 100:.1f}%"
                    validation_set_info = {
                        'total_samples': total_samples,
                        'positive_samples': pos_samples,
                        'negative_samples': neg_samples,
                        'positive_ratio': pos_ratio,
                        'negative_ratio': neg_ratio
                    }
                    
                    # 调试日志
                    print(f"DEBUG: validation_set_info = {validation_set_info}")
                    
                    # 检查是否有正负样本
                    unique_labels = np.unique(true_event_labels)
                    if len(unique_labels) > 1:
                        event_auc = roc_auc_score(true_event_labels, pred_event_probs)
                        event_prauc = average_precision_score(true_event_labels, pred_event_probs)
                        
                        # P2/P3: 在验证阶段进行阈值校准（只在有事件数据时）
                        try:
                            # 确保calibrator已初始化
                            if not hasattr(self, 'threshold_calibrator') or self.threshold_calibrator is None:
                                calibrator_output_dir = os.path.join('outputs', 'calibration')
                                if hasattr(self, 'fold_id'):
                                    calibrator_output_dir = os.path.join('outputs', 'calibration', f'fold_{self.fold_id}')
                                # 对于无正样本的情况，允许使用默认阈值
                                allow_default = (len(unique_labels) <= 1)
                                self.threshold_calibrator = EventThresholdCalibrator(
                                    output_dir=calibrator_output_dir,
                                    allow_default_threshold=allow_default
                                )
                            
                            # 执行校准
                            self.threshold_calibrator.fit(
                                true_event_labels, pred_event_probs, data_split="validation"
                            )
                            # 使用get_metadata()获取完整元数据，而不仅仅是阈值
                            calibrated_metadata = self.threshold_calibrator.get_metadata()
                            calibrated_thresholds = {
                                'threshold_f2_best': calibrated_metadata['threshold_f2_best'],
                                'threshold_strict': calibrated_metadata['threshold_strict'],
                                'threshold_loose': calibrated_metadata['threshold_loose']
                            }
                            threshold_calibration_success = True
                            
                            # 记录成功信息
                            logger.info(f"Threshold calibration successful for fold {getattr(self, 'fold_id', 'main')}: "
                                       f"F2-best={calibrated_thresholds['threshold_f2_best']:.4f}, "
                                       f"strict={calibrated_thresholds['threshold_strict']:.4f}, "
                                       f"loose={calibrated_thresholds['threshold_loose']:.4f}")
                            
                        except Exception as calib_error:
                            logger.warning(f"Threshold calibration failed: {calib_error}")
                            # 如果校准失败，尝试使用默认阈值
                            if len(unique_labels) <= 1:
                                # 验证集只有单类别，使用默认阈值
                                calibrated_thresholds = {
                                    'threshold_f2_best': 0.5,
                                    'threshold_strict': 0.5, 
                                    'threshold_loose': 0.3
                                }
                                # 创建默认的calibrated_metadata
                                calibrated_metadata = {
                                    'threshold_f2_best': 0.5,
                                    'threshold_strict': 0.5,
                                    'threshold_loose': 0.3,
                                    'source_split': 'validation',
                                    'val_pr_auc': event_prauc,
                                    'val_roc_auc': event_auc,
                                    'val_f2_at_threshold_f2_best': 0.0,
                                    'val_recall_at_threshold_f2_best': 0.0 if unique_labels[0] == 0 else 1.0,
                                    'val_fpr_at_threshold_f2_best': 0.0 if unique_labels[0] == 1 else 1.0,
                                    'val_recall_at_threshold_strict': 0.0 if unique_labels[0] == 0 else 1.0,
                                    'val_fpr_at_threshold_strict': 0.0 if unique_labels[0] == 1 else 1.0,
                                    'val_recall_at_threshold_loose': 0.0 if unique_labels[0] == 0 else 1.0,
                                    'val_fpr_at_threshold_loose': 0.0 if unique_labels[0] == 1 else 1.0
                                }
                                threshold_calibration_success = True
                                logger.info("Using default thresholds due to single-class validation set")
                            else:
                                calibrated_thresholds = None
                                calibrated_metadata = None
                                threshold_calibration_success = False
                    else:
                        event_auc = 0.5  # 只有一个类别时AUC为0.5
                        event_prauc = 0.0 if unique_labels[0] == 0 else 1.0
                        logger.warning(f"Validation set has only one class (all {unique_labels[0]}), skipping threshold calibration")
                except Exception as e:
                    logger.warning(f"事件检测AUC/PR-AUC计算失败: {e}")
                    event_auc = 0.0
                    event_prauc = 0.0
            
            # 转换为Python原生类型
            event_auc = float(event_auc)
            event_prauc = float(event_prauc)
            
            # 分阶段计算综合评分 - 使用任务6要求的公式
            mae_normalized = mae / self.persistence_mae_baseline
            current_phase = self.curriculum_scheduler.get_current_phase(epoch)
            
            if current_phase == 1:
                # Phase 1: 只监控val_mae
                val_combined = mae_normalized
            elif current_phase == 2:
                # Phase 2: val_combined = val_mae_ratio - 0.3 * val_yellow_recall
                val_combined = mae_normalized - 0.3 * yellow_recall
            else:
                # Phase 3+: 任务6新指标：val_combined = -0.5*event_auc - 0.3*(1-val_mae_ratio) - 0.2*yellow_recall
                # 注意：越小越好，所以取负号
                val_combined = -0.5 * event_auc - 0.3 * (1 - mae_normalized) - 0.2 * yellow_recall
            
            # 计算多任务综合分数（用于best_multi checkpoint）
            # score_multi = 0.45 * val_disp_mae_ratio + 0.35 * (1 - val_event_pr_auc) + 0.10 * val_strict_fpr_at_operating_threshold + 0.10 * (1 - val_event_recall_at_operating_threshold)
            strict_fpr_at_calibrated = 0.0
            recall_at_calibrated = 0.0
            threshold_f2_best = 0.5  # 默认阈值
            threshold_strict = 0.5   # 默认strict阈值
            
            # 初始化所有校准指标
            val_f2_at_f2_best = 0.0
            val_recall_at_f2_best = 0.0
            val_fpr_at_f2_best = 0.0
            val_recall_at_strict = 0.0
            val_fpr_at_strict = 0.0
            val_recall_at_loose = 0.0
            val_fpr_at_loose = 0.0
            
            if threshold_calibration_success and calibrated_thresholds is not None:
                try:
                    # 使用校准后的阈值计算FPR和recall
                    pred_event_probs = torch.sigmoid(torch.cat(all_pred_event_logits, dim=0)).numpy().flatten()
                    true_event_labels = torch.cat(all_true_event, dim=0).numpy().flatten()
                    
                    # F2-best阈值下的指标
                    pred_binary_f2 = (pred_event_probs >= calibrated_thresholds['threshold_f2_best']).astype(int)
                    tp_f2 = np.sum((pred_binary_f2 == 1) & (true_event_labels == 1))
                    fp_f2 = np.sum((pred_binary_f2 == 1) & (true_event_labels == 0))
                    tn_f2 = np.sum((pred_binary_f2 == 0) & (true_event_labels == 0))
                    fn_f2 = np.sum((pred_binary_f2 == 0) & (true_event_labels == 1))
                    
                    precision_f2 = tp_f2 / (tp_f2 + fp_f2) if (tp_f2 + fp_f2) > 0 else 0.0
                    recall_f2 = tp_f2 / (tp_f2 + fn_f2) if (tp_f2 + fn_f2) > 0 else 0.0
                    val_f2_at_f2_best = 5 * precision_f2 * recall_f2 / (4 * precision_f2 + recall_f2) if (4 * precision_f2 + recall_f2) > 0 else 0.0
                    val_recall_at_f2_best = recall_f2
                    val_fpr_at_f2_best = fp_f2 / (fp_f2 + tn_f2) if (fp_f2 + tn_f2) > 0 else 0.0
                    strict_fpr_at_calibrated = val_fpr_at_f2_best
                    recall_at_calibrated = val_recall_at_f2_best
                    threshold_f2_best = calibrated_thresholds['threshold_f2_best']
                    threshold_strict = calibrated_thresholds['threshold_strict']
                    
                    # Strict阈值下的指标
                    pred_binary_strict = (pred_event_probs >= calibrated_thresholds['threshold_strict']).astype(int)
                    tp_strict = np.sum((pred_binary_strict == 1) & (true_event_labels == 1))
                    fp_strict = np.sum((pred_binary_strict == 1) & (true_event_labels == 0))
                    tn_strict = np.sum((pred_binary_strict == 0) & (true_event_labels == 0))
                    fn_strict = np.sum((pred_binary_strict == 0) & (true_event_labels == 1))
                    
                    val_recall_at_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
                    val_fpr_at_strict = fp_strict / (fp_strict + tn_strict) if (fp_strict + tn_strict) > 0 else 0.0
                    
                    # Loose阈值下的指标
                    pred_binary_loose = (pred_event_probs >= calibrated_thresholds['threshold_loose']).astype(int)
                    tp_loose = np.sum((pred_binary_loose == 1) & (true_event_labels == 1))
                    fp_loose = np.sum((pred_binary_loose == 1) & (true_event_labels == 0))
                    tn_loose = np.sum((pred_binary_loose == 0) & (true_event_labels == 0))
                    fn_loose = np.sum((pred_binary_loose == 0) & (true_event_labels == 1))
                    
                    val_recall_at_loose = tp_loose / (tp_loose + fn_loose) if (tp_loose + fn_loose) > 0 else 0.0
                    val_fpr_at_loose = fp_loose / (fp_loose + tn_loose) if (fp_loose + tn_loose) > 0 else 0.0
                    
                    # 保存完整的校准元数据
                    self._last_validation_calibrated_metadata = calibrated_metadata
                    self._last_validation_threshold_calibration_success = True
                    self._last_validation_calibrated_thresholds = calibrated_thresholds
                    
                except Exception as e:
                    logger.warning(f"FPR/Recall计算失败: {e}")
                    strict_fpr_at_calibrated = 0.0
                    recall_at_calibrated = 0.0
            else:
                # 回退到固定阈值0.5（但应该避免这种情况）
                if len(all_pred_event_logits) > 0 and len(all_true_event) > 0:
                    try:
                        pred_event_probs = torch.sigmoid(torch.cat(all_pred_event_logits, dim=0)).numpy().flatten()
                        true_event_labels = torch.cat(all_true_event, dim=0).numpy().flatten()
                        
                        pred_binary = (pred_event_probs >= 0.5).astype(int)
                        tp = np.sum((pred_binary == 1) & (true_event_labels == 1))
                        fp = np.sum((pred_binary == 1) & (true_event_labels == 0))
                        tn = np.sum((pred_binary == 0) & (true_event_labels == 0))
                        fn = np.sum((pred_binary == 0) & (true_event_labels == 1))
                        
                        strict_fpr_at_calibrated = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        recall_at_calibrated = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        threshold_f2_best = 0.5
                        threshold_strict = 0.5
                    except Exception as e:
                        logger.warning(f"FPR/Recall计算失败: {e}")
                        strict_fpr_at_calibrated = 0.0
                        recall_at_calibrated = 0.0
            
            strict_fpr_at_calibrated = float(strict_fpr_at_calibrated)
            recall_at_calibrated = float(recall_at_calibrated)
            threshold_f2_best = float(threshold_f2_best)
            
            score_multi = (0.45 * mae_normalized + 
                          0.35 * (1 - event_prauc) + 
                          0.10 * strict_fpr_at_calibrated + 
                          0.10 * (1 - recall_at_calibrated))
            score_multi = float(score_multi)
            val_combined = float(val_combined)
            val_monitor_score = score_multi
            
            return {
                'val_mae': mae,
                'val_rmse': rmse,
                'val_r2': r2,
                'val_f1_weighted': f1_weighted,
                'val_f1_macro': f1_macro,
                'val_accuracy': accuracy,
                'val_yellow_recall': yellow_recall,
                'val_red_recall': red_recall,
                'val_event_auc': event_auc,
                'val_event_prauc': event_prauc,
                'val_strict_fpr_at_calibrated': strict_fpr_at_calibrated,
                'val_recall_at_calibrated': recall_at_calibrated,
                'val_threshold_f2_best': threshold_f2_best,
                'val_threshold_strict': threshold_strict,
                'val_calibrated_thresholds': calibrated_thresholds,
                'val_calibrated_metadata': calibrated_metadata if threshold_calibration_success else None,
                # 添加完整的校准指标
                'val_f2_at_f2_best': val_f2_at_f2_best,
                'val_recall_at_f2_best': val_recall_at_f2_best,
                'val_fpr_at_f2_best': val_fpr_at_f2_best,
                'val_recall_at_strict': val_recall_at_strict,
                'val_fpr_at_strict': val_fpr_at_strict,
                'val_recall_at_loose': val_recall_at_loose,
                'val_fpr_at_loose': val_fpr_at_loose,
                'val_score_multi': score_multi,
                'val_combined_score': val_combined,
                'val_monitor_score': val_monitor_score,
                'event_finetune_enabled': self.event_finetune_enabled,
                'validation_set_info': validation_set_info
            }
        except Exception as e:
            print(f"Validation error details: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_combined_score': self.best_combined_score,
            'config': self.config
        }
        
        # 保存最近3个检查点
        recent_checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        recent_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        while len(recent_checkpoints) >= 3:
            oldest = recent_checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
            
    def _apply_phase4_lr_reduction(self):
        """应用Phase 4的学习率降低"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1
        print(f"Phase 4: 学习率降低到 {self._get_current_lr():.2e}")

    def _get_phase_boundary_epochs(self) -> Dict[str, int]:
        """返回课程学习各阶段开始epoch，优先使用config中的custom curriculum。"""
        if getattr(self.curriculum_scheduler, 'use_custom', False):
            stages = list(getattr(self.curriculum_scheduler, 'curriculum_stages', []))
            return {
                'phase2_start': int(stages[0]) if len(stages) >= 1 else 20,
                'phase3_start': int(stages[1]) if len(stages) >= 2 else 60,
                'phase4_start': int(stages[2]) if len(stages) >= 3 else 100,
            }
        return {'phase2_start': 20, 'phase3_start': 60, 'phase4_start': 100}

    def _step_learning_rate(self, epoch: int):
        """正确推进学习率调度。

        原实现只创建了scheduler但从未step，导致LR长期停留在base_lr / warmup_epochs。
        这里采用显式warmup + 原scheduler的组合，确保学习率真正变化。
        """
        if epoch < self.warmup_epochs - 1:
            next_factor = float(epoch + 2) / self.warmup_epochs
            next_lr = self.base_lr * next_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = next_lr
            return

        if isinstance(self.cosine_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            self.cosine_scheduler.step(epoch - self.warmup_epochs + 1)
        else:
            self.cosine_scheduler.step()
    
    def fit(self) -> Dict[str, list]:
        """主训练循环

        修复点：
        1. 每个epoch只做一次validate，避免重复校准/重复写checkpoint。
        2. early stopping与best-metric统一监控phase-invariant的val_monitor_score(=val_score_multi)。
        3. 正确step学习率调度器，避免LR长期固定。
        4. 课程阶段切换日志和patience切换与config中的curriculum_stages保持一致。
        """
        training_history = {
            'epochs': [],
            'train_losses': [],
            'val_metrics': []
        }

        print("Starting training...")
        print(f"Total epochs: {self.config.training.max_epochs}")
        print(f"Min epochs: {self.config.training.min_epochs}")
        print("-" * 100)

        phase_boundaries = self._get_phase_boundary_epochs()
        previous_phase = None

        for epoch in range(self.config.training.max_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch(epoch)

            validate_every_n_epochs = int(getattr(self.config.training, 'validate_every_n_epochs', 1))
            should_validate = (validate_every_n_epochs <= 1) or ((epoch + 1) % validate_every_n_epochs == 0) or (epoch == self.config.training.max_epochs - 1)

            val_metrics = None
            if should_validate:
                val_metrics = self.validate(epoch)

            # 正确推进学习率调度（在epoch结束后更新下一epoch的lr）
            self._step_learning_rate(epoch)
            current_lr = self._get_current_lr()

            # 记录训练指标
            self.writer.add_scalar('train/loss_total', train_loss, epoch)
            if hasattr(self, 'epoch_loss_components'):
                for loss_name, tb_name in [
                    ('loss_disp', 'train/loss_disp'),
                    ('loss_aux', 'train/loss_aux'),
                    ('loss_risk', 'train/loss_risk'),
                    ('loss_event', 'train/loss_event'),
                    ('loss_creep', 'train/loss_creep'),
                    ('loss_stress', 'train/loss_stress'),
                    ('loss_seismic', 'train/loss_seismic'),
                    ('loss_causal', 'train/loss_causal'),
                    ('wloss_disp', 'train/weighted_loss_disp'),
                    ('wloss_aux', 'train/weighted_loss_aux'),
                    ('wloss_risk', 'train/weighted_loss_risk'),
                    ('wloss_event', 'train/weighted_loss_event'),
                    ('wloss_creep', 'train/weighted_loss_creep'),
                    ('wloss_stress', 'train/weighted_loss_stress'),
                    ('wloss_seismic', 'train/weighted_loss_seismic'),
                    ('wloss_causal', 'train/weighted_loss_causal'),
                ]:
                    self.writer.add_scalar(tb_name, self.epoch_loss_components.get(loss_name, 0.0), epoch)
            self.writer.add_scalar('train/lr', current_lr, epoch)

            current_phase = self.curriculum_scheduler.get_current_phase(epoch)
            self.writer.add_scalar('train/phase', current_phase, epoch)
            if current_phase != previous_phase:
                print(f"🎯 Entering {self.curriculum_scheduler.get_phase_name(epoch)} at epoch {epoch}")
                previous_phase = current_phase
                if (not self._phase2_patience_updated) and epoch >= phase_boundaries['phase2_start']:
                    self.early_stopping.update_patience(self.config.training.phase2_patience)
                    self._phase2_patience_updated = True
                    print(f"🔄 Switched early-stopping patience to: {self.config.training.phase2_patience}")
                if epoch == phase_boundaries['phase4_start']:
                    print("🎯 Phase 4开始 - 精细调优")

            should_early_stop = False

            if val_metrics is not None:
                # 验证指标TensorBoard记录
                self.writer.add_scalar('val/mae_mm', val_metrics['val_mae'], epoch)
                val_mae_ratio = val_metrics['val_mae'] / 2.38
                self.writer.add_scalar('val/mae_ratio', val_mae_ratio, epoch)
                self.writer.add_scalar('val/rmse_mm', val_metrics['val_rmse'], epoch)
                self.writer.add_scalar('val/r2', val_metrics['val_r2'], epoch)
                self.writer.add_scalar('val/f1_weighted', val_metrics['val_f1_weighted'], epoch)
                self.writer.add_scalar('val/combined_score', val_metrics['val_combined_score'], epoch)
                self.writer.add_scalar('val/monitor_score', val_metrics['val_monitor_score'], epoch)
                self.writer.add_scalar('val/event_prauc', val_metrics.get('val_event_prauc', 0.0), epoch)
                self.writer.add_scalar('val/strict_fpr', val_metrics.get('val_strict_fpr_at_calibrated', 0.0), epoch)
                self.writer.add_scalar('val/recall_at_calibrated', val_metrics.get('val_recall_at_calibrated', 0.0), epoch)
                if 'val_yellow_recall' in val_metrics:
                    self.writer.add_scalar('val/yellow_recall', val_metrics['val_yellow_recall'], epoch)
                if 'val_red_recall' in val_metrics:
                    self.writer.add_scalar('val/red_recall', val_metrics['val_red_recall'], epoch)

                # 使用phase-invariant monitor metric更新best score
                current_score = val_metrics['val_monitor_score']
                is_best = current_score < self.best_combined_score
                if is_best:
                    self.best_combined_score = current_score
                    self.best_metrics = val_metrics.copy()

                # 事件头微调（保留原逻辑，但只基于单次validate）
                if val_metrics.get('event_finetune_enabled', False) and not hasattr(self, 'event_finetune_completed'):
                    logger.info("Starting event head fine-tuning phase...")
                    print("🎯 Starting event head fine-tuning phase (3-5 epochs)...")
                    for name, param in self.model.named_parameters():
                        if 'event_head' not in name:
                            param.requires_grad = False
                    finetune_epochs = 3
                    original_lr = self._get_current_lr()
                    finetune_lr = original_lr * 0.1
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = finetune_lr
                    for finetune_epoch in range(finetune_epochs):
                        finetune_loss = self.train_epoch(epoch + finetune_epoch + 0.1)
                        finetune_val_metrics = self.validate(epoch + finetune_epoch + 0.1)
                        print(
                            f"  Event head finetune epoch {finetune_epoch+1}/{finetune_epochs}: "
                            f"loss={finetune_loss:.4f}, PR-AUC={finetune_val_metrics['val_event_prauc']:.4f}"
                        )
                    for name, param in self.model.named_parameters():
                        param.requires_grad = True
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = original_lr
                    self.event_finetune_completed = True
                    logger.info("Event head fine-tuning completed!")
                    print("✅ Event head fine-tuning completed!")

                # checkpoint保存逻辑（保留原逻辑，但基于单次validate结果）
                phase_name = self.curriculum_scheduler.get_phase_name(epoch)
                if hasattr(self, '_last_validation_threshold_calibration_success') and self._last_validation_threshold_calibration_success:
                    calibrated_thresholds = self._last_validation_calibrated_thresholds
                    calibrated_metadata = self._last_validation_calibrated_metadata
                    operating_threshold = calibrated_thresholds.get('threshold_f2_best', 0.5)
                    strict_threshold = calibrated_thresholds.get('threshold_strict', 0.5)
                    loose_threshold = calibrated_thresholds.get('threshold_loose', 0.3)
                    logger.info(
                        f"Threshold calibration successful: operating={operating_threshold:.3f}, "
                        f"strict={strict_threshold:.3f}, loose={loose_threshold:.3f}"
                    )
                else:
                    operating_threshold = getattr(self, '_last_validation_threshold_f2_best', 0.5)
                    strict_threshold = 0.5
                    loose_threshold = 0.3
                    calibrated_metadata = None
                    logger.warning("Using default thresholds (threshold calibration not available or failed)")

                calibration_metrics = {
                    'val_threshold_f2_best': operating_threshold,
                    'val_threshold_strict': strict_threshold,
                    'val_threshold_loose': loose_threshold,
                    'threshold_source': 'validation',
                    'val_f2_at_f2_best': calibrated_metadata.get('val_f2_at_threshold_f2_best', 0.0) if calibrated_metadata else 0.0,
                    'val_recall_at_f2_best': calibrated_metadata.get('val_recall_at_threshold_f2_best', 0.0) if calibrated_metadata else 0.0,
                    'val_fpr_at_f2_best': calibrated_metadata.get('val_fpr_at_threshold_f2_best', 0.0) if calibrated_metadata else 0.0,
                    'val_recall_at_strict': calibrated_metadata.get('val_recall_at_threshold_strict', 0.0) if calibrated_metadata else 0.0,
                    'val_fpr_at_strict': calibrated_metadata.get('val_fpr_at_threshold_strict', 0.0) if calibrated_metadata else 0.0,
                    'val_recall_at_loose': calibrated_metadata.get('val_recall_at_threshold_loose', 0.0) if calibrated_metadata else 0.0,
                    'val_fpr_at_loose': calibrated_metadata.get('val_fpr_at_threshold_loose', 0.0) if calibrated_metadata else 0.0,
                }

                def _build_ckpt_metrics():
                    metrics_for_checkpoint = {
                        'val_disp_mae_mm': val_metrics['val_mae'],
                        'val_event_aucroc': val_metrics.get('val_event_auc', 0.0),
                        'val_event_prauc': val_metrics.get('val_event_prauc', 0.0),
                        'val_score_multi': val_metrics.get('val_score_multi', float('inf')),
                    }
                    metrics_for_checkpoint.update(calibration_metrics)
                    if 'validation_set_info' in val_metrics:
                        metrics_for_checkpoint['validation_set_info'] = val_metrics['validation_set_info']
                    return metrics_for_checkpoint

                if self.checkpoint_manager.should_save_best_disp(val_metrics['val_mae']):
                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch, phase_name, _build_ckpt_metrics(), 'best_disp')
                if self.checkpoint_manager.should_save_best_event(val_metrics['val_event_prauc']):
                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch, phase_name, _build_ckpt_metrics(), 'best_event')
                if self.checkpoint_manager.should_save_best_multi(val_metrics['val_score_multi']):
                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch, phase_name, _build_ckpt_metrics(), 'best_multi')

                if epoch == self.config.training.max_epochs - 1:
                    self.checkpoint_manager.save_last_checkpoint(self.model, self.optimizer, epoch, phase_name, _build_ckpt_metrics())

                # 早停：仅在达到最少epoch后，基于phase-invariant monitor metric判断
                if epoch >= self.config.training.min_epochs - 1:
                    self.early_stopping(val_metrics['val_monitor_score'])
                    should_early_stop = self.early_stopping.early_stop
                    if should_early_stop:
                        self.checkpoint_manager.save_last_checkpoint(self.model, self.optimizer, epoch, phase_name, _build_ckpt_metrics())

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:3d} | Phase {current_phase} | LR: {current_lr:.2e} | "
                        f"Train: {train_loss:.4f} | Val MAE: {val_metrics['val_mae']:.3f} mm | "
                        f"Val MAE ratio: {val_mae_ratio:.3f} | Event PR-AUC: {val_metrics.get('val_event_prauc', 0.0):.3f} | "
                        f"Monitor: {val_metrics['val_monitor_score']:.4f}"
                    )
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Phase {current_phase} | LR: {current_lr:.2e} | Train: {train_loss:.4f} | Validation skipped")

            training_history['epochs'].append(epoch)
            training_history['train_losses'].append(train_loss)
            training_history['val_metrics'].append(val_metrics)

            if should_early_stop:
                print(f"⚠️ Early stopping at epoch {epoch} using monitor={self.monitor_metric}")
                break

        self.writer.close()
        print("Training completed!")
        return training_history

    def _log_training_signals(self, epoch: int, batch_losses: Dict[str, float], 
                            pred_event_logits: Optional[torch.Tensor] = None,
                            y_event: Optional[torch.Tensor] = None):
        """
        训练信号审计日志 - 监控事件头的预测分布和梯度
        """
        if pred_event_logits is not None and y_event is not None:
            with torch.no_grad():
                pred_event_prob = torch.sigmoid(pred_event_logits).cpu().numpy()
                y_event_np = y_event.cpu().numpy()
                
                # 计算统计量
                prob_mean = pred_event_prob.mean()
                prob_std = pred_event_prob.std()
                pred_positive_ratio = (pred_event_prob > 0.5).mean()
                true_positive_ratio = (y_event_np == 1).mean()
                
                logger.info(f"Epoch {epoch} - Event Head Training Signals:")
                logger.info(f"  Pred prob mean: {prob_mean:.4f}, std: {prob_std:.4f}")
                logger.info(f"  Pred positive ratio (>0.5): {pred_positive_ratio:.4f}")
                logger.info(f"  True positive ratio: {true_positive_ratio:.4f}")
                logger.info(f"  Event loss: {batch_losses.get('event_loss', 0.0):.4f}")
                
                # 检查异常情况
                if prob_mean > 0.8:
                    logger.warning(f"WARNING: Event head highly biased toward positive predictions (mean={prob_mean:.4f})")
                elif prob_mean < 0.1:
                    logger.warning(f"WARNING: Event head highly biased toward negative predictions (mean={prob_mean:.4f})")

# 简单的单元测试
if __name__ == "__main__":
    print("This module is designed to be used with the full PI-PHM pipeline.")
    print("Please use it with proper data loaders and model configuration.")