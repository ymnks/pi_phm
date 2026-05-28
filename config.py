"""PI-PHM 配置文件 - 集中管理所有超参数"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml
from datetime import datetime


@dataclass
class DataConfig:
    # 数据路径配置
    data_dir: str = "/home/lab1111/zy/2024_AknesLandslide_Aspaas"  # 数据集根目录路径
    
    # 时间范围配置
    train_end: str = "2020-12-31"      # 训练集结束日期
    val_start: str = "2021-01-15"      # 验证集开始日期  
    val_end: str = "2021-12-15"        # 验证集结束日期（修改：提前16天）
    test_start: str = "2022-01-01"     # 测试集开始日期
    
    # 已知加速事件列表（7个事件的起止日期）
    acceleration_events: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("2011-03-01", "2011-06-30"),   # 2011年春季加速事件
        ("2013-09-01", "2013-12-31"),   # 2013年秋季加速事件  
        ("2015-03-01", "2015-06-30"),   # 2015年春季加速事件
        ("2017-06-01", "2017-09-30"),   # 2017年夏季加速事件
        ("2018-06-01", "2018-09-30"),   # 2018年夏季加速事件
        ("2020-03-01", "2020-06-30"),   # 2020年春季加速事件
        ("2022-03-01", "2022-06-30")    # 2022年春季加速事件
    ])
    
    # 辅助目标列（用于多任务学习）
    auxiliary_targets: List[str] = field(default_factory=lambda: [
        'KH0206_Displacement', 'KH0112_Displacement', 'KH0117_Displacement',
        'KH0118_Displacement', 'KH0217_Displacement', 'KH0218_Displacement', 'KH0306_Displacement'
    ])
    
    # 主要目标列
    target_col: str = "GNSS_12H"


@dataclass
class ModelConfig:
    # 模型结构参数
    d_model: int = 128                 # 模型维度，影响模型容量和计算复杂度
    n_heads: int = 8                   # 注意力头数，用于PatchTST的多头注意力机制
    n_patchtst_layers: int = 2         # PatchTST编码器层数，控制局部时序特征提取深度
    n_mamba_layers: int = 2            # Mamba层数，控制全局序列建模能力
    patch_len: int = 10                # Patch长度，将时间序列分段处理的窗口大小
    stride: int = 5                    # Patch步长，控制相邻patch的重叠程度
    lookback: int = 60                 # 回看窗口长度（天），输入序列的历史长度
    forecast: int = 7                  # 预测窗口长度（天），未来预测的天数
    
    # 静态地质参数（来自 Aspaas 2024 论文及 NGU 报告）
    rock_type: str = "gneiss"          # 岩性：片麻岩
    avg_slope: float = 35.0            # 平均坡度：35°（上部滑体38°，下部28°）
    joint_dip_direction: float = 125.0 # 主控节理倾向：125°
    joint_dip_angle: float = 35.0      # 主控节理倾角：35°
    cohesion: float = 50.0             # 粘聚力 c：50 kPa（反演中值，范围0-100）
    friction_angle: float = 31.0       # 内摩擦角 φ：31°（反演中值，范围29-35）
    rock_density: float = 2700.0       # 岩体密度：2700 kg/m³
    
    # Mamba变体选择（任务2新增）
    mamba_variant: str = "fixed"       # 可选: "fixed"(修复后manual), "official", "gru"
    
    # 课程学习四阶段的epoch边界和各损失权重
    curriculum_stages: List[int] = field(default_factory=lambda: [50, 100, 150, 200])
    stage_weights: List[dict] = field(default_factory=lambda: [
        {"alpha_risk": 0.1, "lambda_creep": 0.01, "lambda_stress": 0.005, "lambda_seismic": 0.005},  # 第一阶段：数据驱动为主
        {"alpha_risk": 0.5, "lambda_creep": 0.05, "lambda_stress": 0.025, "lambda_seismic": 0.025},  # 第二阶段：平衡数据和物理
        {"alpha_risk": 1.0, "lambda_creep": 0.1, "lambda_stress": 0.05, "lambda_seismic": 0.05},      # 第三阶段：完整物理约束
        {"alpha_risk": 1.0, "lambda_creep": 0.1, "lambda_stress": 0.05, "lambda_seismic": 0.05}       # 第四阶段：维持完整约束
    ])


@dataclass
class TrainingConfig:
    # 训练参数
    batch_size: int = 32               # 批次大小，平衡内存使用和梯度稳定性
    max_epochs: int = 200              # 最大训练轮数
    lr: float = 1e-4                   # 学习率，控制参数更新步长
    weight_decay: float = 1e-5         # 权重衰减，L2正则化系数防止过拟合
    patience: int = 20                 # 早停耐心值，验证损失连续多少轮不改善时停止训练
    
    # 新增任务E要求的参数
    min_epochs: int = 60               # 最小训练轮数，确保课程学习充分展开
    phase1_patience: int = 30          # Phase 1 (epoch < 30) 的早停耐心值
    phase2_patience: int = 20          # Phase 2+ (epoch >= 30) 的早停耐心值
    grad_clip: float = 1.0             # 梯度裁剪阈值，防止梯度爆炸
    
    # 损失权重配置
    alpha_risk: float = 1.0            # 风险等级分类损失权重
    lambda_creep: float = 0.1          # 蠕变物理约束损失权重
    lambda_stress: float = 0.05        # 应力平衡物理约束损失权重  
    lambda_seismic: float = 0.05       # 微震活动物理约束损失权重
    
    # FocalLoss alpha权重（对应[GREEN, BLUE, YELLOW, RED]）
    focal_alpha: Optional[List[float]] = None
    
    # 课程学习四阶段的epoch边界和各损失权重
    curriculum_stages: List[int] = field(default_factory=lambda: [50, 100, 150, 200])
    stage_weights: List[dict] = field(default_factory=lambda: [
        {"alpha_risk": 0.1, "lambda_creep": 0.01, "lambda_stress": 0.005, "lambda_seismic": 0.005},  # 第一阶段：数据驱动为主
        {"alpha_risk": 0.5, "lambda_creep": 0.05, "lambda_stress": 0.025, "lambda_seismic": 0.025},  # 第二阶段：平衡数据和物理
        {"alpha_risk": 1.0, "lambda_creep": 0.1, "lambda_stress": 0.05, "lambda_seismic": 0.05},      # 第三阶段：完整物理约束
        {"alpha_risk": 1.0, "lambda_creep": 0.1, "lambda_stress": 0.05, "lambda_seismic": 0.05}       # 第四阶段：维持完整约束
    ])


@dataclass
class PI_PHM_Config:
    """PI-PHM 完整配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig) 
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42                     # 随机种子，确保结果可重现
    label_fusion_method: str = "max_fusion"  # 风险标签融合方法，默认为max_fusion
    event_aware_split: bool = False    # 是否使用事件感知划分
    model_type: str = "full_pip_hm"    # 消融实验模型类型标识
    
    def __post_init__(self):
        """后初始化处理"""
        # 确保日期格式正确
        self._validate_dates()
    
    def _validate_dates(self):
        """验证日期配置的合理性"""
        try:
            train_end_dt = datetime.strptime(self.data.train_end, "%Y-%m-%d")
            val_start_dt = datetime.strptime(self.data.val_start, "%Y-%m-%d")
            val_end_dt = datetime.strptime(self.data.val_end, "%Y-%m-%d")
            test_start_dt = datetime.strptime(self.data.test_start, "%Y-%m-%d")
            
            assert train_end_dt < val_start_dt, "训练结束日期必须早于验证开始日期"
            assert val_start_dt <= val_end_dt, "验证开始日期不能晚于验证结束日期"
            assert val_end_dt < test_start_dt, "验证结束日期必须早于测试开始日期"
        except ValueError as e:
            raise ValueError(f"日期格式错误: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PI_PHM_Config":
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 递归转换字典为dataclass实例
        def dict_to_dataclass(data_dict, dataclass_type):
            if not isinstance(data_dict, dict):
                return data_dict
            
            field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
            kwargs = {}
            for field_name, field_value in data_dict.items():
                if field_name in field_types:
                    field_type = field_types[field_name]
                    if hasattr(field_type, '__dataclass_fields__'):
                        # 嵌套dataclass
                        kwargs[field_name] = dict_to_dataclass(field_value, field_type)
                    else:
                        kwargs[field_name] = field_value
            return dataclass_type(**kwargs)
        
        return dict_to_dataclass(config_dict, cls)
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        import json
        
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        
        config_dict = dataclass_to_dict(self)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

# 为向后兼容性提供Config别名
Config = PI_PHM_Config