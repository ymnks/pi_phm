"""工具函数模块 - 包含通用的辅助函数和类"""
import random
import numpy as np
import torch
from typing import Dict, List, Optional


def set_seed(seed: int = 42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """获取可用的计算设备（GPU/CPU）"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience: int = 7, min_delta: float = 0.0, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss  # 越小越好，所以取负值
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
    def update_patience(self, new_patience):
        """Update the patience value and reset the counter"""
        self.patience = new_patience
        self.counter = 0


class MetricTracker:
    """指标跟踪器类"""
    def __init__(self):
        # TODO: 后续填充具体实现
        pass
    
    def update(self, metrics: Dict[str, float]):
        # TODO: 后续填充具体实现
        pass
    
    def get_best(self, metric_name: str) -> float:
        # TODO: 后续填充具体实现
        return 0.0