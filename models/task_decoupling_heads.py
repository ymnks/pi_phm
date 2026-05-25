class EventDetectionHead(nn.Module):
    def __init__(self, input_dim: int, forecast_horizon: int = 7, pos_ratio: Optional[float] = None):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
        # 全连接层
        self.fc = nn.Linear(input_dim, forecast_horizon)
        
        # 初始化偏置
        if pos_ratio is not None and pos_ratio > 0:
            bias_init = -math.log((1 - pos_ratio) / pos_ratio)
            nn.init.constant_(self.fc.bias, bias_init)
            print(f"EventDetectionHead initialized with pos_ratio={pos_ratio:.4f}, bias_init={bias_init:.4f}")
        else:
            print(f"EventDetectionHead initialized with default bias (pos_ratio={pos_ratio})")
        
        # 创建损失函数
        if pos_ratio is not None and pos_ratio > 0:
            pos_weight = (1 - pos_ratio) / pos_ratio
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
            print(f"EventDetectionHead loss_fn created with pos_weight={pos_weight:.4f} (pos_ratio={pos_ratio:.4f})")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            print(f"EventDetectionHead loss_fn created with default pos_weight=1.0 (pos_ratio={pos_ratio})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)