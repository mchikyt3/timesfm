"""
Training configuration for TimesFM-ICF fine-tuning.

This module provides configuration classes for fine-tuning ICF models
on specific tasks and datasets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os


@dataclass
class ICFTrainingConfig:
    """Configuration for ICF model fine-tuning.
    
    Controls all aspects of the training process including optimization,
    data handling, and model saving.
    """
    
    # Basic training parameters
    num_epochs: int = 10
    """Number of training epochs"""
    
    batch_size: int = 32
    """Batch size for training"""
    
    val_batch_size: Optional[int] = None
    """Validation batch size (defaults to batch_size)"""
    
    learning_rate: float = 1e-4
    """Learning rate for optimizer"""
    
    weight_decay: float = 0.01
    """Weight decay for regularization"""
    
    # Advanced training parameters
    warmup_steps: int = 0
    """Number of warmup steps for learning rate schedule"""
    
    gradient_clip_norm: Optional[float] = 1.0
    """Gradient clipping norm (None to disable)"""
    
    accumulation_steps: int = 1
    """Number of gradient accumulation steps"""
    
    # Task-specific learning rates
    base_model_lr: Optional[float] = None
    """Learning rate for base TimesFM parameters (defaults to learning_rate / 10)"""
    
    icf_components_lr: Optional[float] = None
    """Learning rate for ICF components (defaults to learning_rate)"""
    
    # Model freezing settings
    freeze_base_model: bool = True
    """Whether to freeze base TimesFM parameters"""
    
    freeze_base_embeddings: bool = True
    """Whether to freeze base model embeddings specifically"""
    
    unfreeze_after_epoch: Optional[int] = None
    """Epoch after which to unfreeze base model (None to keep frozen)"""
    
    # Task configuration
    primary_task: str = 'forecasting'
    """Primary task for training ('forecasting', 'classification', etc.)"""
    
    task_weights: Dict[str, float] = field(default_factory=lambda: {'forecasting': 1.0, 'classification': 1.0})
    """Weights for different tasks in multi-task learning"""
    
    multi_task_strategy: str = 'weighted_sum'
    """Multi-task learning strategy: 'weighted_sum', 'alternating', 'gradnorm'"""
    
    # Data configuration  
    max_sequence_length: Optional[int] = None
    """Maximum sequence length (overrides model config if provided)"""
    
    data_shuffle: bool = True
    """Whether to shuffle training data"""
    
    pin_memory: bool = True
    """Whether to pin memory for data loading"""
    
    num_workers: int = 4
    """Number of data loading workers"""
    
    # Validation and early stopping
    validation_frequency: int = 1
    """Validation frequency (every N epochs)"""
    
    early_stopping_patience: int = 5
    """Early stopping patience (epochs without improvement)"""
    
    early_stopping_metric: str = 'val_loss'
    """Metric to monitor for early stopping"""
    
    early_stopping_mode: str = 'min'
    """Early stopping mode: 'min' or 'max'"""
    
    # Checkpointing
    save_frequency: int = 1
    """Model saving frequency (every N epochs)"""
    
    checkpoint_dir: str = './checkpoints'
    """Directory to save model checkpoints"""
    
    save_best_only: bool = True
    """Whether to save only the best model"""
    
    keep_last_n_checkpoints: int = 3
    """Number of recent checkpoints to keep"""
    
    # Logging and monitoring
    log_frequency: int = 100
    """Logging frequency (every N steps)"""
    
    use_wandb: bool = False
    """Whether to use Weights & Biases logging"""
    
    wandb_project: Optional[str] = None
    """W&B project name"""
    
    wandb_entity: Optional[str] = None
    """W&B entity name"""
    
    experiment_name: Optional[str] = None
    """Name for this training experiment"""
    
    # Hardware configuration
    device: str = 'auto'
    """Device for training: 'auto', 'cpu', 'cuda', 'cuda:0', etc."""
    
    mixed_precision: bool = True
    """Whether to use mixed precision training (fp16)"""
    
    compile_model: bool = True
    """Whether to compile the model for training (torch.compile)"""
    
    # Advanced optimization
    optimizer_type: str = 'adamw'
    """Optimizer type: 'adamw', 'adam', 'sgd'"""
    
    scheduler_type: str = 'cosine'
    """Learning rate scheduler: 'cosine', 'linear', 'constant', 'polynomial'"""
    
    beta1: float = 0.9
    """Adam beta1 parameter"""
    
    beta2: float = 0.999
    """Adam beta2 parameter"""
    
    epsilon: float = 1e-8
    """Adam epsilon parameter"""
    
    # Validation and post-processing
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Set default validation batch size
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        
        # Set default learning rates for different components
        if self.base_model_lr is None:
            self.base_model_lr = self.learning_rate / 10  # Lower LR for pretrained components
        
        if self.icf_components_lr is None:
            self.icf_components_lr = self.learning_rate
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Validate task configuration
        valid_tasks = {'forecasting', 'classification', 'regression', 'anomaly', 'generation'}
        if self.primary_task not in valid_tasks:
            raise ValueError(f"Invalid primary_task '{self.primary_task}'. Valid tasks: {valid_tasks}")
        
        for task in self.task_weights.keys():
            if task not in valid_tasks:
                raise ValueError(f"Invalid task in task_weights: '{task}'. Valid tasks: {valid_tasks}")
        
        # Validate multi-task strategy
        valid_strategies = {'weighted_sum', 'alternating', 'gradnorm'}
        if self.multi_task_strategy not in valid_strategies:
            raise ValueError(f"Invalid multi_task_strategy '{self.multi_task_strategy}'. Valid: {valid_strategies}")
        
        # Validate optimizer and scheduler
        valid_optimizers = {'adamw', 'adam', 'sgd'}
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(f"Invalid optimizer_type '{self.optimizer_type}'. Valid: {valid_optimizers}")
        
        valid_schedulers = {'cosine', 'linear', 'constant', 'polynomial'}
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"Invalid scheduler_type '{self.scheduler_type}'. Valid: {valid_schedulers}")
        
        # Validate early stopping
        valid_modes = {'min', 'max'}
        if self.early_stopping_mode not in valid_modes:
            raise ValueError(f"Invalid early_stopping_mode '{self.early_stopping_mode}'. Valid: {valid_modes}")
    
    def get_optimizer_config(self) -> Dict:
        """Get optimizer configuration.
        
        Returns:
            Dictionary with optimizer settings
        """
        return {
            'type': self.optimizer_type,
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': (self.beta1, self.beta2),
            'eps': self.epsilon,
        }
    
    def get_scheduler_config(self) -> Dict:
        """Get learning rate scheduler configuration.
        
        Returns:
            Dictionary with scheduler settings
        """
        return {
            'type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'num_training_steps': None,  # Will be set during training
        }
    
    def get_data_config(self) -> Dict:
        """Get data loading configuration.
        
        Returns:
            Dictionary with data loading settings
        """
        return {
            'batch_size': self.batch_size,
            'val_batch_size': self.val_batch_size,
            'shuffle': self.data_shuffle,
            'pin_memory': self.pin_memory,
            'num_workers': self.num_workers,
        }
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration.
        
        Returns:
            Dictionary with logging settings
        """
        return {
            'log_frequency': self.log_frequency,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'experiment_name': self.experiment_name,
        }


@dataclass
class ICFDatasetConfig:
    """Configuration for ICF dataset preparation.
    
    Controls how data is processed and formatted for ICF training.
    """
    
    # Data paths
    train_data_path: str = ""
    """Path to training data"""
    
    val_data_path: Optional[str] = None
    """Path to validation data (None for train/val split)"""
    
    test_data_path: Optional[str] = None
    """Path to test data"""
    
    # Data splitting
    val_split: float = 0.2
    """Validation split ratio (if val_data_path not provided)"""
    
    test_split: float = 0.1
    """Test split ratio (if test_data_path not provided)"""
    
    random_seed: int = 42
    """Random seed for data splitting"""
    
    # Data processing
    normalize_data: bool = True
    """Whether to normalize time series data"""
    
    normalization_method: str = 'standardize'
    """Normalization method: 'standardize', 'minmax', 'robust'"""
    
    handle_missing_values: str = 'interpolate'
    """Missing value handling: 'interpolate', 'drop', 'forward_fill'"""
    
    # Time series specific
    context_length: int = 128
    """Length of context window for time series"""
    
    prediction_horizon: int = 24
    """Prediction horizon for forecasting tasks"""
    
    stride: int = 1
    """Stride for sliding window extraction"""
    
    # Covariate handling
    covariate_columns: List[str] = field(default_factory=list)
    """List of covariate column names"""
    
    categorical_covariates: List[str] = field(default_factory=list)
    """List of categorical covariate columns"""
    
    # Classification specific
    class_column: Optional[str] = None
    """Column name for classification labels"""
    
    class_mapping: Dict[str, int] = field(default_factory=dict)
    """Mapping from class names to integer labels"""
    
    # Data augmentation
    enable_augmentation: bool = False
    """Whether to enable data augmentation"""
    
    noise_level: float = 0.01
    """Noise level for data augmentation"""
    
    time_warping: bool = False
    """Whether to apply time warping augmentation"""
    
    def __post_init__(self):
        """Validate dataset configuration."""
        # Validate split ratios
        if not 0.0 <= self.val_split <= 1.0:
            raise ValueError("val_split must be between 0.0 and 1.0")
        
        if not 0.0 <= self.test_split <= 1.0:
            raise ValueError("test_split must be between 0.0 and 1.0")
        
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        
        # Validate normalization method
        valid_methods = {'standardize', 'minmax', 'robust'}
        if self.normalization_method not in valid_methods:
            raise ValueError(f"Invalid normalization_method '{self.normalization_method}'. Valid: {valid_methods}")
        
        # Validate missing value handling
        valid_handling = {'interpolate', 'drop', 'forward_fill'}
        if self.handle_missing_values not in valid_handling:
            raise ValueError(f"Invalid handle_missing_values '{self.handle_missing_values}'. Valid: {valid_handling}")


# Predefined training configurations
class ICFTrainingConfigs:
    """Collection of predefined training configurations."""
    
    @staticmethod
    def quick_test() -> ICFTrainingConfig:
        """Configuration for quick testing."""
        return ICFTrainingConfig(
            num_epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            validation_frequency=1,
            save_frequency=1,
            log_frequency=10,
            compile_model=False,
        )
    
    @staticmethod
    def forecasting_finetuning() -> ICFTrainingConfig:
        """Configuration for forecasting fine-tuning."""
        return ICFTrainingConfig(
            num_epochs=20,
            batch_size=32,
            learning_rate=5e-5,
            primary_task='forecasting',
            freeze_base_model=True,
            unfreeze_after_epoch=10,
            early_stopping_patience=5,
            scheduler_type='cosine',
        )
    
    @staticmethod
    def classification_training() -> ICFTrainingConfig:
        """Configuration for classification training."""
        return ICFTrainingConfig(
            num_epochs=15,
            batch_size=64,
            learning_rate=1e-4,
            primary_task='classification',
            task_weights={'classification': 1.0},
            freeze_base_model=True,
            gradient_clip_norm=1.0,
        )
    
    @staticmethod
    def multi_task_training() -> ICFTrainingConfig:
        """Configuration for multi-task training."""
        return ICFTrainingConfig(
            num_epochs=25,
            batch_size=32,
            learning_rate=3e-5,
            primary_task='forecasting',
            task_weights={'forecasting': 0.7, 'classification': 0.3},
            multi_task_strategy='weighted_sum',
            freeze_base_model=False,
            unfreeze_after_epoch=5,
        )
    
    @staticmethod
    def production_training() -> ICFTrainingConfig:
        """Configuration for production training."""
        return ICFTrainingConfig(
            num_epochs=50,
            batch_size=64,
            learning_rate=2e-5,
            warmup_steps=1000,
            gradient_clip_norm=0.5,
            early_stopping_patience=10,
            save_best_only=True,
            mixed_precision=True,
            use_wandb=True,
        )