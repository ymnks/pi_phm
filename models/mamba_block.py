import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# 尝试导入mamba-ssm官方库
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    
    class MambaEncoderOfficial(nn.Module):
        """官方mamba-ssm实现（方案X2）"""
        
        def __init__(self, config: 'PI_PHM_Config'):
            super().__init__()
            self.d_model = config.model.d_model
            self.n_layers = config.model.n_mamba_layers
            self.d_state = getattr(config.model, 'd_state', 16)
            self.d_conv = getattr(config.model, 'd_conv', 4)
            self.expand = getattr(config.model, 'expand', 2)
            
            # 创建LayerNorm + Mamba层
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    Mamba(
                        d_model=self.d_model,
                        d_state=self.d_state,
                        d_conv=self.d_conv,
                        expand=self.expand,
                    )
                ) for _ in range(self.n_layers)
            ])
            
            # 创建对应的MLP层
            self.mlp_layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model * 2),
                    nn.GELU(),
                    nn.Linear(self.d_model * 2, self.d_model),
                ) for _ in range(self.n_layers)
            ])
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
            """
            Args:
                x: (B, n_patches, d_model)
            Returns:
                output: (B, n_patches, d_model)
                ssm_states: None (官方实现不直接提供SSM状态)
            """
            B, n_patches, d_model = x.shape
            assert d_model == self.d_model, f"Input dim {d_model} != expected {self.d_model}"
            
            for attn, mlp in zip(self.layers, self.mlp_layers):
                x = x + attn(x)
                x = x + mlp(x)
                
            return x, None
            
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    MambaEncoderOfficial = None
    print("Warning: mamba-ssm not available, official implementation disabled")


class SelectiveSSM(nn.Module):
    """手动实现的简化版Mamba块（备选方案）"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 8, bias=False)  # B, C, Δ
        self.dt_proj = nn.Linear(8, self.d_inner, bias=True)  # Δ projection
        
        # A矩阵（状态转移矩阵）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
        
        # 初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Δ偏置初始化
        dt_init_std = self.d_inner**-0.5 * math.log(2)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        # A矩阵初始化为负值确保稳定性
        # 修复：确保A_log被正确初始化
        with torch.no_grad():
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log.copy_(torch.log(A))
            
    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, 
                      B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        手动实现选择性扫描（递推形式）
        Args:
            u: (B, L, d_inner)
            delta: (B, L, d_inner)
            A: (d_inner, d_state)
            B: (B, L, d_state)
            C: (B, L, d_state)
        Returns:
            y: (B, L, d_inner)
        """
        B_, L, d_inner = u.shape
        d_state = self.d_state
        
        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)
        
        # 递推计算状态
        h = torch.zeros(B_, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = (h @ C[:, i].unsqueeze(-1)).squeeze(-1)  # (B, d_inner)
            ys.append(y)
            
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
            ssm_states: (B, d_inner, d_state) - 最后一个时间步的状态
        """
        B, L, d_model = x.shape
        assert d_model == self.d_model, f"Input dim {d_model} != expected {self.d_model}"
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner), (B, L, d_inner)
        
        # 因果卷积
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B, L, d_inner)
        x_conv = nn.functional.silu(x_conv)
        
        # SSM参数生成
        x_dbl = self.x_proj(x_conv)  # (B, L, d_state + d_state + 8)
        delta = self.dt_proj(x_dbl[:, :, -8:])  # (B, L, d_inner)
        delta = nn.functional.softplus(delta)
        B_param = x_dbl[:, :, :self.d_state]  # (B, L, d_state)
        C_param = x_dbl[:, :, self.d_state:self.d_state*2]  # (B, L, d_state)
        
        # 获取A矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 选择性扫描
        y_ssm = self.selective_scan(x_conv, delta, A, B_param, C_param)  # (B, L, d_inner)
        
        # 添加残差连接D
        y_ssm = y_ssm + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # 输出
        y = y_ssm * nn.functional.silu(z)
        output = self.out_proj(y)  # (B, L, d_model)
        
        # 保存最后一个时间步的状态用于可视化
        ssm_states = None  # 在手动实现中难以提取完整状态，返回None
        
        return output, ssm_states


class MambaBlock(nn.Module):
    """单个Mamba块（支持官方库和手动实现）"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        
        # 使用官方mamba-ssm实现（如果可用）
        if MAMBA_SSM_AVAILABLE and torch.cuda.is_available():
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_official = True
            print("Note: Using official mamba-ssm implementation")
        else:
            # 手动实现
            self.mamba = SelectiveSSM(d_model, d_state, d_conv, expand)
            self.use_official = False
            if MAMBA_SSM_AVAILABLE:
                print("Note: mamba-ssm available but using manual implementation (CPU mode)")
            else:
                print("Note: Using manual Mamba implementation (mamba-ssm not available)")
            
        # LayerNorm和MLP
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
            ssm_states: (B, d_inner, d_state) or None
        """
        # Pre-Norm + Mamba + Residual
        x_norm1 = self.norm1(x)
        if self.use_official:
            # 官方库不直接提供SSM状态，返回None
            mamba_out = self.mamba(x_norm1)
            ssm_states = None
        else:
            mamba_out, ssm_states = self.mamba(x_norm1)
            
        x = x + mamba_out
        
        # Pre-Norm + MLP + Residual
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        output = x + mlp_out
        
        return output, ssm_states


class MambaEncoder(nn.Module):
    """Mamba编码器主类，支持多种实现方案"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        """
        Args:
            config: PI-PHM配置对象，需包含config.model.mamba_variant参数
        """
        super().__init__()
        self.d_model = config.model.d_model
        self.n_layers = config.model.n_mamba_layers
        
        # 获取Mamba变体配置，默认为"fixed"
        mamba_variant = getattr(config.model, 'mamba_variant', 'fixed')
        print(f"DEBUG: mamba_variant from config: '{mamba_variant}'")
        
        # 根据配置选择实现方案
        if mamba_variant == "official" and MambaEncoderOfficial is not None:
            print(f"Using official mamba-ssm implementation (variant: {mamba_variant})")
            self.encoder = MambaEncoderOfficial(config)
        elif mamba_variant == "gru":
            print(f"Using GRU implementation (variant: {mamba_variant})")
            self.encoder = MambaEncoderGRU(config)
        else:
            print(f"Using fixed manual Mamba implementation (variant: {mamba_variant})")
            self.encoder = FixedManualMamba(config)
            
    def forward(self, h_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_patches: (B, n_patches, d_model) - PatchTST输出，n_patches=11
            
        Returns:
            h_mamba: (B, n_patches, d_model) - 编码后的表示
            states: 最后一层的状态（取决于具体实现）
        """
        return self.encoder(h_patches)
    
    def get_gradient_norms(self):
        """获取各层参数的梯度范数"""
        norms = {}
        for i, layer in enumerate(self.layers):
            mamba_norm = 0.0
            mlp_norm = 0.0
            
            # Mamba部分梯度范数
            for name, param in layer.mamba.named_parameters():
                if param.grad is not None:
                    mamba_norm += param.grad.norm().item()
            
            # MLP部分梯度范数  
            for name, param in layer.mlp.named_parameters():
                if param.grad is not None:
                    mlp_norm += param.grad.norm().item()
                    
            norms[f'layer_{i}_mamba'] = mamba_norm
            norms[f'layer_{i}_mlp'] = mlp_norm
            
        return norms


class FixedManualMamba(nn.Module):
    """修复后的manual Mamba实现（方案X1）"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        """
        Args:
            config: PI-PHM配置对象
        """
        super().__init__()
        self.d_model = config.model.d_model
        self.n_layers = config.model.n_mamba_layers
        self.d_state = getattr(config.model, 'd_state', 16)  # SSM隐状态维度
        self.d_conv = getattr(config.model, 'd_conv', 4)    # 因果卷积核大小
        self.expand = getattr(config.model, 'expand', 2)    # 内部扩展因子
        
        # Mamba块堆叠
        self.layers = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand)
            for _ in range(self.n_layers)
        ])
        
    def forward(self, h_patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_patches: (B, n_patches, d_model) - PatchTST输出，n_patches=11
            
        Returns:
            h_mamba: (B, n_patches, d_model) - Mamba编码后的表示
            ssm_states: (B, d_inner, d_state) or None - 最后一层的SSM状态
        """
        B, n_patches, d_model = h_patches.shape
        assert d_model == self.d_model, f"Input dim {d_model} != expected {self.d_model}"
        
        h_mamba = h_patches
        final_ssm_states = None
        
        for layer in self.layers:
            h_mamba, ssm_states = layer(h_mamba)
            final_ssm_states = ssm_states  # 保存最后一层的状态
            
        return h_mamba, final_ssm_states
    
    def get_gradient_norms(self):
        """获取各层参数的梯度范数"""
        norms = {}
        for i, layer in enumerate(self.layers):
            mamba_norm = 0.0
            mlp_norm = 0.0
            
            # Mamba部分梯度范数
            for name, param in layer.mamba.named_parameters():
                if param.grad is not None:
                    mamba_norm += param.grad.norm().item()
            
            # MLP部分梯度范数  
            for name, param in layer.mlp.named_parameters():
                if param.grad is not None:
                    mlp_norm += param.grad.norm().item()
                    
            norms[f'layer_{i}_mamba'] = mamba_norm
            norms[f'layer_{i}_mlp'] = mlp_norm
            
        return norms


class MambaEncoderGRU(nn.Module):
    """GRU替代Mamba（方案X3）"""
    
    def __init__(self, config: 'PI_PHM_Config'):
        super().__init__()
        self.d_model = config.model.d_model
        self.n_layers = config.model.n_mamba_layers
        
        self.norm1 = nn.LayerNorm(self.d_model)
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.1 if self.n_layers > 1 else 0,
            bidirectional=False,  # 保持因果性
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.output_proj = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_patches, d_model)
        Returns:
            output: (B, n_patches, d_model)
            hidden: (num_layers, B, d_model) - GRU隐藏状态
        """
        B, n_patches, d_model = x.shape
        assert d_model == self.d_model, f"Input dim {d_model} != expected {self.d_model}"
        
        residual = x
        x_norm = self.norm1(x)
        gru_out, hidden = self.gru(x_norm)
        x = residual + self.output_proj(gru_out)
        x = x + self.mlp(self.norm2(x))
        return x, hidden


# 单元测试
if __name__ == "__main__":
    import sys
    import os
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from config import PI_PHM_Config
    
    # 创建配置
    config = PI_PHM_Config()
    
    # 测试参数
    B, n_patches, d_model = 2, 11, 128
    
    # 创建随机输入
    h_patches = torch.randn(B, n_patches, d_model)
    
    # 测试MambaEncoder
    encoder = MambaEncoder(config)
    h_mamba, ssm_states = encoder(h_patches)
    
    print(f"Input shape: {h_patches.shape}")
    print(f"Output shape: {h_mamba.shape}")
    print(f"SSM states: {ssm_states}")
    print(f"Using official mamba-ssm: {MAMBA_SSM_AVAILABLE and hasattr(encoder.layers[0], 'use_official') and encoder.layers[0].use_official}")
    
    # 验证输出形状
    assert h_mamba.shape == (B, n_patches, d_model), f"Output shape error: {h_mamba.shape} != {(B, n_patches, d_model)}"
    
    # 测试梯度回传
    loss = h_mamba.sum()
    if ssm_states is not None:
        loss = loss + ssm_states.sum()
    loss.backward()
    print("Gradient computation successful!")
    
    print("All tests passed!")