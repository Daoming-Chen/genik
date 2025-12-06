"""Training Configuration for GeNIK.

This module provides configuration dataclasses and utilities for training
the GeNIK inverse kinematics network.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import json

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Attributes
    ----------
    hidden_size : int
        Size of hidden layers.
    num_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to use batch normalization.
    use_residual_output : bool
        Whether to use residual connection from input to output.
    model_type : str
        Model type: 'standard' or 'large'.
    """
    hidden_size: int = 256
    num_blocks: int = 4
    dropout: float = 0.1
    use_batch_norm: bool = False
    use_residual_output: bool = True
    model_type: str = 'standard'


@dataclass
class OptimizerConfig:
    """Optimizer configuration.
    
    Attributes
    ----------
    name : str
        Optimizer name: 'adamw', 'adam', 'sgd'.
    learning_rate : float
        Base learning rate.
    weight_decay : float
        Weight decay (L2 regularization).
    betas : tuple
        Adam beta parameters.
    momentum : float
        SGD momentum.
    """
    name: str = 'adamw'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.
    
    Attributes
    ----------
    name : str
        Scheduler name: 'cosine', 'step', 'plateau', 'none'.
    warmup_epochs : int
        Number of warmup epochs.
    warmup_start_lr : float
        Starting learning rate for warmup.
    min_lr : float
        Minimum learning rate for cosine annealing.
    step_size : int
        Step size for StepLR.
    gamma : float
        Decay factor for StepLR and ReduceLROnPlateau.
    patience : int
        Patience for ReduceLROnPlateau.
    """
    name: str = 'cosine'
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-5
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10


@dataclass
class LossConfig:
    """Loss function configuration.
    
    Attributes
    ----------
    joint_loss_type : str
        Joint loss type: 'mse' or 'l1'.
    lambda_min : float
        Minimum pose loss weight.
    lambda_max : float
        Maximum pose loss weight.
    lambda_warmup_epochs : int
        Warmup epochs for lambda scheduling.
    w_pos : float
        Weight for position error.
    w_ori : float
        Weight for orientation error.
    """
    joint_loss_type: str = 'mse'
    lambda_min: float = 0.1
    lambda_max: float = 1.0
    lambda_warmup_epochs: int = 50
    w_pos: float = 1.0
    w_ori: float = 1.0


@dataclass
class DataConfig:
    """Data configuration.
    
    Attributes
    ----------
    train_path : str
        Path to training data.
    val_path : str
        Path to validation data (optional, will split from train if not provided).
    test_path : str
        Path to test data (optional).
    batch_size : int
        Training batch size.
    val_batch_size : int
        Validation batch size (defaults to batch_size).
    num_workers : int
        Number of data loading workers.
    train_ratio : float
        Training set ratio when splitting.
    val_ratio : float
        Validation set ratio when splitting.
    normalize_pose : bool
        Whether to normalize pose inputs.
    """
    train_path: str = ''
    val_path: str = ''
    test_path: str = ''
    batch_size: int = 256
    val_batch_size: int = 0  # 0 means same as batch_size
    num_workers: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    normalize_pose: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration.
    
    Attributes
    ----------
    urdf_path : str
        Path to robot URDF file.
    end_link : str
        Name of end-effector link (optional).
    output_dir : str
        Output directory for checkpoints and logs.
    experiment_name : str
        Name of the experiment.
    num_epochs : int
        Number of training epochs.
    gradient_clip : float
        Gradient clipping threshold (0 to disable).
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    val_every_n_epochs : int
        Validation frequency.
    save_every_n_epochs : int
        Checkpoint saving frequency.
    early_stopping_patience : int
        Early stopping patience (0 to disable).
    seed : int
        Random seed.
    device : str
        Device: 'cuda', 'cpu', or 'auto'.
    use_amp : bool
        Whether to use automatic mixed precision.
    model : ModelConfig
        Model configuration.
    optimizer : OptimizerConfig
        Optimizer configuration.
    scheduler : SchedulerConfig
        Scheduler configuration.
    loss : LossConfig
        Loss configuration.
    data : DataConfig
        Data configuration.
    """
    # Paths
    urdf_path: str = ''
    end_link: str = ''
    output_dir: str = 'outputs'
    experiment_name: str = 'genik'
    
    # Training
    num_epochs: int = 100
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 20
    
    # System
    seed: int = 42
    device: str = 'auto'
    use_amp: bool = False
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set val_batch_size if not specified
        if self.data.val_batch_size == 0:
            self.data.val_batch_size = self.data.batch_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        # Handle nested configs
        if 'model' in d and isinstance(d['model'], dict):
            d['model'] = ModelConfig(**d['model'])
        if 'optimizer' in d and isinstance(d['optimizer'], dict):
            d['optimizer'] = OptimizerConfig(**d['optimizer'])
        if 'scheduler' in d and isinstance(d['scheduler'], dict):
            d['scheduler'] = SchedulerConfig(**d['scheduler'])
        if 'loss' in d and isinstance(d['loss'], dict):
            d['loss'] = LossConfig(**d['loss'])
        if 'data' in d and isinstance(d['data'], dict):
            d['data'] = DataConfig(**d['data'])
        
        return cls(**d)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file.
        
        Parameters
        ----------
        path : str or Path
            Output file path (.yaml or .json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        d = self.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(d, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(d, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from file.
        
        Parameters
        ----------
        path : str or Path
            Input file path (.yaml or .json).
            
        Returns
        -------
        TrainingConfig
            Loaded configuration.
        """
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                d = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                d = json.load(f)
        
        return cls.from_dict(d)
    
    def get_experiment_dir(self) -> Path:
        """Get experiment output directory."""
        return Path(self.output_dir) / self.experiment_name


def create_default_config(
    urdf_path: str,
    data_path: str,
    output_dir: str = 'outputs',
) -> TrainingConfig:
    """Create a default configuration.
    
    Parameters
    ----------
    urdf_path : str
        Path to robot URDF file.
    data_path : str
        Path to training data.
    output_dir : str
        Output directory.
        
    Returns
    -------
    TrainingConfig
        Default configuration.
    """
    return TrainingConfig(
        urdf_path=urdf_path,
        output_dir=output_dir,
        data=DataConfig(train_path=data_path),
    )
