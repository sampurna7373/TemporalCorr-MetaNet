"""Default configuration for TemporalCorr-MetaNet.

Contains all hyperparameters from the technical document Appendix A.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone
    in_channels: int = 3
    backbone_channels: List[int] = field(default_factory=lambda: [32, 48, 64])
    
    # PTCF Module
    ptcf_out_channels: int = 256
    ptcf_reduction: int = 4
    
    # ATSM Module
    num_classes: int = 2  # change, no-change
    
    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    
    # Input size
    input_size: Tuple[int, int] = (512, 512)


@dataclass
class MetaLearningConfig:
    """Meta-learning configuration from Appendix A."""
    # Learning rates
    meta_lr: float = 0.001  # α: outer loop learning rate
    inner_lr: float = 0.01  # β: inner loop learning rate
    
    # Episode configuration
    num_classes: int = 2  # M: number of classes per episode
    support_samples: int = 5  # K: support samples per class
    query_samples: int = 15  # N: query samples per class
    
    # Training
    max_episodes: int = 10000
    val_interval: int = 500
    
    # Optimization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Inner loop steps
    inner_loop_steps: int = 5


@dataclass
class LossConfig:
    """Loss function configuration."""
    # Focal Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Positive class weight for imbalance
    positive_weight: float = 20.0
    
    # Auxiliary loss weights
    query_loss_weight: float = 0.5
    correlation_sparsity_weight: float = 0.001
    boundary_smoothness_weight: float = 0.0005
    uncertainty_reg_weight: float = 0.0002


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths (to be overridden per dataset)
    data_root: str = "./data"
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Preprocessing
    image_size: Tuple[int, int] = (512, 512)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    flip_prob: float = 0.5
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    contrast_range: Tuple[float, float] = (-0.1, 0.1)
    noise_std_range: Tuple[float, float] = (0.001, 0.005)
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Device
    device: str = "cuda"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 1000
    
    # Logging
    log_dir: str = "./logs"
    log_freq: int = 100
    
    # Early stopping
    patience: int = 20  # In terms of validation intervals
    
    # Learning rate schedule
    lr_scheduler: str = "exponential"
    lr_gamma: float = 0.9
    lr_step_size: int = 1000
    warmup_episodes: int = 500
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Complete configuration for TemporalCorr-MetaNet."""
    model: ModelConfig = field(default_factory=ModelConfig)
    meta: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment name
    experiment_name: str = "temporalcorr_metanet"
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "meta": self.meta.__dict__,
            "loss": self.loss.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "experiment_name": self.experiment_name,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                setattr(config.model, k, v)
        if "meta" in config_dict:
            for k, v in config_dict["meta"].items():
                setattr(config.meta, k, v)
        if "loss" in config_dict:
            for k, v in config_dict["loss"].items():
                setattr(config.loss, k, v)
        if "data" in config_dict:
            for k, v in config_dict["data"].items():
                setattr(config.data, k, v)
        if "training" in config_dict:
            for k, v in config_dict["training"].items():
                setattr(config.training, k, v)
        if "experiment_name" in config_dict:
            config.experiment_name = config_dict["experiment_name"]
        
        return config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def merge_config_with_args(config: Config, args) -> Config:
    """Merge configuration with command line arguments."""
    # Override with command line arguments if provided
    if hasattr(args, "data_root") and args.data_root:
        config.data.data_root = args.data_root
    if hasattr(args, "device") and args.device:
        config.training.device = args.device
    if hasattr(args, "max_episodes") and args.max_episodes:
        config.meta.max_episodes = args.max_episodes
    if hasattr(args, "seed") and args.seed:
        config.training.seed = args.seed
    if hasattr(args, "checkpoint_dir") and args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if hasattr(args, "log_dir") and args.log_dir:
        config.training.log_dir = args.log_dir
    
    return config
