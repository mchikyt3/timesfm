"""
Configuration classes for TimesFM-ICF models.

This module provides dataclass-based configuration for ICF model initialization
and behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ICFConfig:
    """Configuration for TimesFM-ICF model initialization.
    
    Controls the behavior and architecture of the ICF wrapper around
    the base TimesFM model.
    """
    
    # Task configuration
    enabled_tasks: List[str] = field(default_factory=lambda: ['forecasting'])
    """List of tasks to enable: ['forecasting', 'classification', 'regression', 'anomaly']"""
    
    # Template configuration for sequence building
    max_examples: int = 4
    """Maximum number of examples in context window"""
    
    max_covariates_per_example: int = 3
    """Maximum number of covariate tokens per example"""
    
    max_ts_tokens_per_example: int = 10
    """Maximum number of time series tokens per example"""
    
    max_class_tokens_per_example: int = 1
    """Maximum number of classification tokens per example"""
    
    total_sequence_length: int = 64
    """Fixed total sequence length for torch.compile compatibility"""
    
    # Model architecture
    base_hidden_size: int = 1280
    """Hidden size of base TimesFM model"""
    
    num_separator_tokens: int = 10
    """Number of separator token embeddings to learn"""
    
    num_classification_classes: int = 4
    """Number of classification classes (uptrend, downtrend, stable, volatile)"""
    
    dropout_rate: float = 0.1
    """Dropout rate for ICF components"""
    
    # Compilation settings
    enable_torch_compile: bool = True
    """Whether to enable torch.compile optimization"""
    
    compile_mode: str = "max-autotune"
    """torch.compile mode: 'default', 'reduce-overhead', 'max-autotune'"""
    
    compile_fullgraph: bool = True
    """Whether to compile the full graph (recommended for fixed templates)"""
    
    # Tokenization settings
    time_series_vocab_range: tuple = (1000, 49999)
    """Token ID range for time series values (avoiding special tokens)"""
    
    covariate_vocab_range: tuple = (1000, 49999)  
    """Token ID range for covariate values"""
    
    # Integration settings
    freeze_base_model: bool = True
    """Whether to freeze base TimesFM parameters during ICF training"""
    
    use_base_model_tokenizer: bool = False
    """Whether to use TimesFM's native tokenizer (if available)"""
    
    # Validation
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate enabled tasks
        valid_tasks = {'forecasting', 'classification', 'regression', 'anomaly', 'generation'}
        for task in self.enabled_tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task '{task}'. Valid tasks: {valid_tasks}")
        
        # Validate sequence length configuration
        min_seq_length = self._calculate_min_sequence_length()
        if self.total_sequence_length < min_seq_length:
            raise ValueError(f"total_sequence_length ({self.total_sequence_length}) too small. "
                           f"Minimum required: {min_seq_length}")
        
        # Validate compile mode
        valid_modes = {'default', 'reduce-overhead', 'max-autotune'}
        if self.compile_mode not in valid_modes:
            raise ValueError(f"Invalid compile_mode '{self.compile_mode}'. Valid modes: {valid_modes}")
        
        # Validate vocab ranges
        if self.time_series_vocab_range[0] >= self.time_series_vocab_range[1]:
            raise ValueError("time_series_vocab_range must be (min, max) with min < max")
        if self.covariate_vocab_range[0] >= self.covariate_vocab_range[1]:
            raise ValueError("covariate_vocab_range must be (min, max) with min < max")
    
    def _calculate_min_sequence_length(self) -> int:
        """Calculate minimum sequence length needed for template."""
        slots_per_example = (
            self.max_covariates_per_example +  # covariate slots
            1 +  # covariate separator
            self.max_ts_tokens_per_example +   # time series slots
            (self.max_class_tokens_per_example + 1 if 'classification' in self.enabled_tasks else 0) +  # class slots + sep
            1    # example separator (except last)
        )
        
        min_length = slots_per_example * self.max_examples - 1  # -1 for no separator after last example
        return min_length
    
    def get_template_config(self) -> Dict:
        """Get template configuration for sequence builder.
        
        Returns:
            Dictionary with template configuration
        """
        return {
            'max_examples': self.max_examples,
            'max_covariates_per_example': self.max_covariates_per_example,
            'max_ts_tokens_per_example': self.max_ts_tokens_per_example,
            'max_class_tokens_per_example': self.max_class_tokens_per_example,
            'total_sequence_length': self.total_sequence_length,
        }
    
    def get_compilation_config(self) -> Dict:
        """Get torch.compile configuration.
        
        Returns:
            Dictionary with compilation settings
        """
        return {
            'enabled': self.enable_torch_compile,
            'mode': self.compile_mode,
            'fullgraph': self.compile_fullgraph,
        }


@dataclass
class ICFModelState:
    """State information for initialized ICF model.
    
    Tracks the current state and configuration of an ICF model instance.
    """
    
    # Model state
    is_initialized: bool = False
    """Whether the model has been fully initialized"""
    
    base_model_loaded: bool = False
    """Whether the base TimesFM model has been loaded"""
    
    icf_components_initialized: bool = False
    """Whether ICF components have been initialized"""
    
    is_compiled: bool = False
    """Whether the model has been compiled with torch.compile"""
    
    # Configuration tracking
    loaded_config: Optional[ICFConfig] = None
    """Configuration used to initialize the model"""
    
    enabled_tasks: List[str] = field(default_factory=list)
    """Currently enabled tasks"""
    
    base_model_info: Dict = field(default_factory=dict)
    """Information about the loaded base model"""
    
    # Performance tracking
    compilation_time: Optional[float] = None
    """Time taken for model compilation (seconds)"""
    
    last_inference_time: Optional[float] = None
    """Time for last inference call (seconds)"""
    
    def mark_base_model_loaded(self, model_info: Dict):
        """Mark base model as loaded with info."""
        self.base_model_loaded = True
        self.base_model_info = model_info
        self._update_initialization_state()
    
    def mark_icf_components_initialized(self, config: ICFConfig):
        """Mark ICF components as initialized."""
        self.icf_components_initialized = True
        self.loaded_config = config
        self.enabled_tasks = config.enabled_tasks.copy()
        self._update_initialization_state()
    
    def mark_compiled(self, compilation_time: float):
        """Mark model as compiled."""
        self.is_compiled = True
        self.compilation_time = compilation_time
    
    def _update_initialization_state(self):
        """Update overall initialization state."""
        self.is_initialized = (
            self.base_model_loaded and 
            self.icf_components_initialized
        )
    
    def get_summary(self) -> Dict:
        """Get summary of model state.
        
        Returns:
            Dictionary with model state summary
        """
        return {
            'is_initialized': self.is_initialized,
            'base_model_loaded': self.base_model_loaded,
            'icf_components_initialized': self.icf_components_initialized,
            'is_compiled': self.is_compiled,
            'enabled_tasks': self.enabled_tasks,
            'compilation_time': self.compilation_time,
            'base_model_info': self.base_model_info,
        }


# Predefined configurations for common use cases
class ICFConfigs:
    """Collection of predefined ICF configurations."""
    
    @staticmethod
    def forecasting_only() -> ICFConfig:
        """Configuration for forecasting-only tasks."""
        return ICFConfig(
            enabled_tasks=['forecasting'],
            max_examples=4,
            max_ts_tokens_per_example=12,
            max_covariates_per_example=4,
            total_sequence_length=80,
        )
    
    @staticmethod
    def classification_only() -> ICFConfig:
        """Configuration for classification-only tasks."""
        return ICFConfig(
            enabled_tasks=['classification'],
            max_examples=3,
            max_ts_tokens_per_example=8,
            max_covariates_per_example=2,
            max_class_tokens_per_example=1,
            total_sequence_length=64,
            num_classification_classes=4,
        )
    
    @staticmethod
    def multi_task() -> ICFConfig:
        """Configuration for multi-task learning."""
        return ICFConfig(
            enabled_tasks=['forecasting', 'classification'],
            max_examples=3,
            max_ts_tokens_per_example=10,
            max_covariates_per_example=3,
            max_class_tokens_per_example=1,
            total_sequence_length=96,
            num_classification_classes=4,
        )
    
    @staticmethod
    def large_context() -> ICFConfig:
        """Configuration for larger context windows."""
        return ICFConfig(
            enabled_tasks=['forecasting', 'classification'],
            max_examples=6,
            max_ts_tokens_per_example=15,
            max_covariates_per_example=5,
            max_class_tokens_per_example=1,
            total_sequence_length=160,
            num_classification_classes=4,
        )
    
    @staticmethod
    def debug() -> ICFConfig:
        """Configuration for debugging (small, fast)."""
        return ICFConfig(
            enabled_tasks=['forecasting'],
            max_examples=2,
            max_ts_tokens_per_example=4,
            max_covariates_per_example=2,
            total_sequence_length=32,
            enable_torch_compile=False,  # Disable compilation for debugging
        )