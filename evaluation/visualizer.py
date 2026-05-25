import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import Dict, List, Optional
from .evaluator import PIPHMEvaluator

class PIPHMVisualizer:
    """PI-PHM模型可视化器"""
    
    def __init__(self, config):
        self.config = config
        self.colors = {
            'GREEN': '#2ecc71',
            'BLUE': '#3498db', 
            'YELLOW': '#f1c40f',
            'RED': '#e74c3c'
        }
        self.risk_labels = ['GREEN', 'BLUE', 'YELLOW', 'RED']
        
        # 设置绘图风格
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': (12, 8),
            'savefig.dpi': 300,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
        
        # 创建输出目录
        os.makedirs('outputs/figures', exist_ok=True)
    
    def visualize_all(self, evaluator: PIPHMEvaluator, metrics: Dict, 
                     preds: np.ndarray, trues: np.ndarray, 
                     pred_risk: np.ndarray, true_risk: np.ndarray,
                     timestamps: List[str], gate_info: List[Dict], 
                     attn_weights: List[np.ndarray],
                     training_history: Optional[Dict] = None):
        """生成所有可视化图表"""
        # 转换timestamps为datetime
        dt_timestamps = pd.to_datetime(timestamps)
        
        # 图1: 位移预测时间序列图
        self._plot_displacement_prediction(dt_timestamps, preds, trues)
        
        # 图2: 风险等级预测时间序列图
        self._plot_risk_prediction(dt_timestamps, pred_risk, true_risk)
        
        # 图3: 混淆矩阵
        self._plot_confusion_matrix(true_risk, pred_risk)
        
        # 图4: 各步长MAE柱状图
        self._plot_stepwise_mae(metrics['displacement']['stepwise_mae'])
        
        # 图5: Attention权重可视化
        self._plot_attention_weights(attn_weights, pred_risk, dt_timestamps)
        
        # 图6: Physics Gate门控权重分析
        self._plot_physics_gate_analysis(gate_info, pred_risk)
        
        # 图7: 训练历史曲线
        if training_history is not None:
            self._plot_training_history(training_history)
        
        # 图8: 加速事件案例分析
        self._plot_event_case_study(dt_timestamps, preds, trues, pred_risk, true_risk)
    
    def _plot_displacement_prediction(self, timestamps: pd.DatetimeIndex, 
                                    preds: np.ndarray, trues: np.ndarray):
        """绘制位移预测时间序列图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 上半部分：真实值 vs 预测值
        ax1.plot(timestamps, trues[:, 0], 'k-', linewidth=1.5, label='True Displacement')
        ax1.plot(timestamps, preds[:, 0], 'r--', linewidth=1.5, label='Predicted Displacement')
        
        # 添加置信带（简化：使用预测的标准差）
        pred_std = np.std(preds, axis=1)
        ax1.fill_between(timestamps, 
                        preds[:, 0] - pred_std, 
                        preds[:, 0] + pred_std, 
                        color='red', alpha=0.2, label='±1σ Confidence')
        
        # 标注已知加速事件区域
        known_events = [
            {'start': '2022-05-01', 'end': '2022-07-15'}
        ]
        for event in known_events:
            start = pd.to_datetime(event['start'])
            end = pd.to_datetime(event['end'])
            ax1.axvspan(start, end, color='yellow', alpha=0.3, label='Known Acceleration Event')
        
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title('Displacement Prediction Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下半部分：预测误差
        errors = preds[:, 0] - trues[:, 0]
        ax2.plot(timestamps, errors, 'b-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Error (mm)')
        ax2.set_title('Prediction Error Time Series')
        ax2.grid(True, alpha=0.3)
        
        # 标注加速事件区域
        for event in known_events:
            start = pd.to_datetime(event['start'])
            end = pd.to_datetime(event['end'])
            ax2.axvspan(start, end, color='yellow', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/displacement_prediction.png', bbox_inches='tight')
        plt.close()
    
    def _plot_risk_prediction(self, timestamps: pd.DatetimeIndex, 
                             pred_risk: np.ndarray, true_risk: np.ndarray):
        """绘制风险等级预测时间序列图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 上半部分：真实标签 vs 预测标签（彩色条带）
        colors_true = [self.colors[self.risk_labels[r]] for r in true_risk]
        colors_pred = [self.colors[self.risk_labels[r]] for r in pred_risk]
        
        # 创建颜色映射的y值
        y_true = np.ones(len(timestamps)) * 0.8
        y_pred = np.ones(len(timestamps)) * 0.2
        
        ax1.scatter(timestamps, y_true, c=colors_true, s=20, marker='s', label='True Risk')
        ax1.scatter(timestamps, y_pred, c=colors_pred, s=20, marker='s', label='Predicted Risk')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors[label], label=label) 
                          for label in self.risk_labels]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_title('Risk Level Prediction')
        ax1.grid(True, alpha=0.3)
        
        # 下半部分：各类别的预测概率曲线（这里简化，实际需要从模型输出获取概率）
        # 由于我们只有最终预测结果，这里用one-hot编码近似
        pred_probs = np.zeros((len(pred_risk), 4))
        for i, r in enumerate(pred_risk):
            pred_probs[i, r] = 1.0
        
        for i, label in enumerate(self.risk_labels):
            ax2.plot(timestamps, pred_probs[:, i], 
                    color=self.colors[label], linewidth=1, label=f'{label} Probability')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Probability')
        ax2.set_title('Risk Level Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 标注加速事件区域
        known_events = [
            {'start': '2022-05-01', 'end': '2022-07-15'}
        ]
        for event in known_events:
            start = pd.to_datetime(event['start'])
            end = pd.to_datetime(event['end'])
            ax2.axvspan(start, end, color='yellow', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/risk_prediction.png', bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, true_risk: np.ndarray, pred_risk: np.ndarray):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_risk, pred_risk, labels=[0, 1, 2, 3])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=self.risk_labels, yticklabels=self.risk_labels,
                   ax=ax)
        ax.set_xlabel('Predicted Risk Level')
        ax.set_ylabel('True Risk Level')
        ax.set_title('Confusion Matrix (Normalized by Row)')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/confusion_matrix.png', bbox_inches='tight')
        plt.close()
    
    def _plot_stepwise_mae(self, stepwise_mae: List[float]):
        """绘制各步长MAE柱状图"""
        days = [f'Day{i+1}' for i in range(len(stepwise_mae))]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(days, stepwise_mae, color='steelblue', alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('MAE (mm)')
        ax.set_title('Stepwise MAE by Prediction Day')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/stepwise_mae.png', bbox_inches='tight')
        plt.close()
    
    def _plot_attention_weights(self, attn_weights: List[np.ndarray], 
                               pred_risk: np.ndarray, timestamps: pd.DatetimeIndex):
        """绘制Attention权重可视化"""
        # 选取3个代表性样本
        stable_idx = np.where(pred_risk == 0)[0][:1]  # GREEN样本
        accel_idx = np.where(pred_risk == 2)[0][:1]   # YELLOW样本  
        high_risk_idx = np.where(pred_risk == 3)[0][:1]  # RED样本
        
        selected_indices = []
        if len(stable_idx) > 0:
            selected_indices.append(('Stable Period', stable_idx[0]))
        if len(accel_idx) > 0:
            selected_indices.append(('Acceleration Period', accel_idx[0]))
        if len(high_risk_idx) > 0:
            selected_indices.append(('High Risk Period', high_risk_idx[0]))
        
        if not selected_indices:
            # 如果没有找到特定样本，选择前3个
            selected_indices = [('Sample 1', 0), ('Sample 2', 1), ('Sample 3', 2)]
        
        n_samples = min(len(selected_indices), 3)
        if n_samples == 0:
            return  # 没有可用的样本
            
        fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
        
        if n_samples == 1:
            axes = [axes]
        
        for i, (title, idx) in enumerate(selected_indices[:n_samples]):
            if idx < len(attn_weights):
                attn = attn_weights[idx]
                
                # 处理不同形状的注意力权重
                if attn.ndim == 4:
                    # 形状为 (batch, heads, seq_len, seq_len) - 取第一个样本的第一个head
                    attn = attn[0, 0]  # (seq_len, seq_len)
                elif attn.ndim == 3:
                    # 形状为 (heads, seq_len, seq_len) - 取第一个head
                    attn = attn[0]  # (seq_len, seq_len)
                elif attn.ndim == 2:
                    # 已经是2D数组，直接使用
                    pass
                else:
                    # 其他形状：跳过这个样本
                    print(f"Warning: Unexpected attention weight shape {attn.shape}, skipping sample {idx}")
                    continue
                
                # 确保attn是2D数组
                if attn.ndim != 2:
                    print(f"Warning: Failed to convert attention weight to 2D, shape: {attn.shape}")
                    continue
                    
                im = axes[i].imshow(attn, cmap='viridis', aspect='auto')
                axes[i].set_title(f'{title}\n{timestamps[idx].strftime("%Y-%m-%d")}')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('outputs/figures/attention_weights.png', bbox_inches='tight')
        plt.close()
    
    def _plot_physics_gate_analysis(self, gate_info: List[Dict], pred_risk: np.ndarray):
        """绘制Physics Gate门控权重分析"""
        # 提取维度门控权重
        g_dim_list = []
        g_patch_list = []
        risk_levels = []
        
        for i, info in enumerate(gate_info):
            if 'g_dim' in info and info['g_dim'] is not None:
                g_dim = info['g_dim'].squeeze()
                if g_dim.ndim == 1:
                    g_dim_list.append(g_dim)
                    risk_levels.append(pred_risk[i])
                    
                    if 'g_patch' in info and info['g_patch'] is not None:
                        g_patch = info['g_patch'].squeeze()
                        if g_patch.ndim == 1:
                            g_patch_list.append(g_patch)
        
        if not g_dim_list:
            return
            
        g_dim_array = np.array(g_dim_list)
        g_patch_array = np.array(g_patch_list) if g_patch_list else None
        
        # 图1: 维度门控权重分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按风险等级分组绘制维度门控
        for risk_level in [0, 1, 2, 3]:
            mask = np.array(risk_levels) == risk_level
            if np.any(mask):
                avg_g_dim = np.mean(g_dim_array[mask], axis=0)
                ax1.plot(avg_g_dim, label=f'{self.risk_labels[risk_level]}', 
                        color=self.colors[self.risk_labels[risk_level]])
        
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Gate Weight')
        ax1.set_title('Dimension-wise Gate Weights by Risk Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: Patch门控在不同风险等级下的模式
        if g_patch_array is not None:
            for risk_level in [0, 1, 2, 3]:
                mask = np.array(risk_levels) == risk_level
                if np.any(mask):
                    avg_g_patch = np.mean(g_patch_array[mask], axis=0)
                    ax2.plot(avg_g_patch, label=f'{self.risk_labels[risk_level]}',
                            color=self.colors[self.risk_labels[risk_level]])
            
            ax2.set_xlabel('Patch Position')
            ax2.set_ylabel('Gate Weight')
            ax2.set_title('Patch-wise Gate Weights by Risk Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No patch gate data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/physics_gate_analysis.png', bbox_inches='tight')
        plt.close()
    
    def _plot_training_history(self, training_history: Dict):
        """绘制训练历史曲线"""
        epochs = range(len(training_history['train_loss']))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练/验证损失
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in training_history:
            ax1.plot(epochs, training_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 验证MAE
        if 'val_mae' in training_history:
            ax2.plot(epochs, training_history['val_mae'], 'g-')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (mm)')
            ax2.set_title('Validation MAE')
            ax2.grid(True, alpha=0.3)
        
        # 验证F1
        if 'val_f1' in training_history:
            ax3.plot(epochs, training_history['val_f1'], 'm-')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('Validation F1 Score')
            ax3.grid(True, alpha=0.3)
        
        # 学习率
        if 'learning_rate' in training_history:
            ax4.plot(epochs, training_history['learning_rate'], 'c-')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            
            # 标注课程学习阶段边界
            phase_boundaries = [30, 80, 150]  # Phase 1-2, 2-3, 3-4的边界
            for boundary in phase_boundaries:
                if boundary < len(epochs):
                    ax4.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
                    ax4.text(boundary, ax4.get_ylim()[1]*0.9, f'Phase {boundary//30+1}-{boundary//30+2}', 
                            rotation=90, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/training_history.png', bbox_inches='tight')
        plt.close()
    
    def _plot_event_case_study(self, timestamps: pd.DatetimeIndex, 
                              preds: np.ndarray, trues: np.ndarray,
                              pred_risk: np.ndarray, true_risk: np.ndarray):
        """绘制加速事件案例分析"""
        # 找到测试集中的加速事件
        event_start = pd.to_datetime('2022-05-01')
        event_end = pd.to_datetime('2022-07-15')
        
        event_mask = (timestamps >= event_start) & (timestamps <= event_end)
        event_indices = np.where(event_mask)[0]
        
        if len(event_indices) == 0:
            return
            
        # 选取事件期间的数据
        event_timestamps = timestamps[event_indices]
        event_preds = preds[event_indices]
        event_trues = trues[event_indices]
        event_pred_risk = pred_risk[event_indices]
        event_true_risk = true_risk[event_indices]
        
        # 计算速率和倒速率
        velocities = np.diff(event_trues[:, 0])
        inv_velocities = 1.0 / (np.abs(velocities) + 1e-8)
        
        # 创建多子图
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # 位移
        axes[0].plot(event_timestamps[:-1], event_trues[:-1, 0], 'k-', label='True Displacement')
        axes[0].plot(event_timestamps[:-1], event_preds[:-1, 0], 'r--', label='Predicted Displacement')
        axes[0].set_ylabel('Displacement (mm)')
        axes[0].set_title('Displacement During Acceleration Event')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 速率
        axes[1].plot(event_timestamps[1:], velocities, 'b-', label='Velocity')
        axes[1].set_ylabel('Velocity (mm/day)')
        axes[1].set_title('Velocity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 倒速率
        axes[2].plot(event_timestamps[1:], inv_velocities, 'g-', label='Inverse Velocity')
        axes[2].set_ylabel('Inverse Velocity (day/mm)')
        axes[2].set_title('Inverse Velocity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 风险等级
        colors_pred = [self.colors[self.risk_labels[r]] for r in event_pred_risk[:-1]]
        axes[3].scatter(event_timestamps[:-1], np.ones(len(event_timestamps)-1)*0.5, 
                       c=colors_pred, s=20, marker='s')
        axes[3].set_ylabel('Risk Level')
        axes[3].set_yticks([])
        axes[3].set_title('Predicted Risk Level')
        axes[3].grid(True, alpha=0.3)
        
        # 添加预警时间点标注
        blue_or_higher = np.where(event_pred_risk[:-1] >= 1)[0]
        if len(blue_or_higher) > 0:
            first_warning_idx = blue_or_higher[0]
            warning_date = event_timestamps[first_warning_idx]
            axes[3].axvline(x=warning_date, color='purple', linestyle='--', 
                           label=f'First Warning ({warning_date.strftime("%Y-%m-%d")})')
            axes[3].legend()
        
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/event_case_study.png', bbox_inches='tight')
        plt.close()