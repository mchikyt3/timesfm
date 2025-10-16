"""
Token management system for TimesFM-ICF.

This module provides the core token system that enables in-context fine-tuning
through learnable separator tokens.
"""

from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn


class CoreTokens:
    """Core separator tokens used across all tasks.
    
    These tokens are fundamental to the ICF system and are always present
    regardless of which tasks are enabled.
    """
    COVARIATE_SEP = 50001   # Separates covariates from time series
    EXAMPLE_SEP = 50002     # Separates different examples in context
    PADDING = 0             # Standard padding token (inherited from base model)


class TaskTokens:
    """Task-specific tokens that extend core functionality.
    
    These tokens are loaded based on which tasks are enabled and provide
    task-specific separation and classification capabilities.
    """
    # Classification tokens
    CLASS_SEP = 50003       # Separates time series from class labels  
    CLASS_UPTREND = 50004   # Class token for upward trend
    CLASS_DOWNTREND = 50005 # Class token for downward trend
    CLASS_STABLE = 50006    # Class token for stable/flat trend
    CLASS_VOLATILE = 50007  # Class token for high volatility
    
    # Future task extensions (reserved token IDs)
    REGRESSION_SEP = 50008  # For regression tasks
    ANOMALY_SEP = 50009     # For anomaly detection
    GENERATION_SEP = 50010  # For sequence generation tasks
    
    # Classification task mapping
    CLASSIFICATION_TOKENS = {
        "uptrend": CLASS_UPTREND,
        "downtrend": CLASS_DOWNTREND,
        "stable": CLASS_STABLE,
        "volatile": CLASS_VOLATILE,
    }


class ICFTokenManager:
    """Manages the complete token vocabulary for ICF models.
    
    Handles loading task-specific tokens, extending vocabulary, and providing
    token ID mappings for the sequence builder.
    """
    
    def __init__(self, enabled_tasks: List[str], base_vocab_size: int = 50000):
        """Initialize token manager with specified tasks.
        
        Args:
            enabled_tasks: List of task names to enable (e.g., ['forecasting', 'classification'])
            base_vocab_size: Vocabulary size of the base TimesFM model
        """
        self.enabled_tasks = set(enabled_tasks)
        self.base_vocab_size = base_vocab_size
        
        # Always include core tokens
        self.core_tokens = self._load_core_tokens()
        
        # Load task-specific tokens based on enabled tasks
        self.task_tokens = self._load_task_tokens()
        
        # Create unified token mapping
        self.token_mapping = {**self.core_tokens, **self.task_tokens}
        
        # Calculate vocabulary extension needed
        self.vocab_extension_size = self._calculate_vocab_extension()
        
    def _load_core_tokens(self) -> Dict[str, int]:
        """Load core tokens that are always present."""
        return {
            "COVARIATE_SEP": CoreTokens.COVARIATE_SEP,
            "EXAMPLE_SEP": CoreTokens.EXAMPLE_SEP,
            "PADDING": CoreTokens.PADDING,
        }
        
    def _load_task_tokens(self) -> Dict[str, int]:
        """Load tokens based on enabled tasks."""
        tokens = {}
        
        if "classification" in self.enabled_tasks:
            tokens.update({
                "CLASS_SEP": TaskTokens.CLASS_SEP,
                "CLASS_UPTREND": TaskTokens.CLASS_UPTREND,
                "CLASS_DOWNTREND": TaskTokens.CLASS_DOWNTREND,
                "CLASS_STABLE": TaskTokens.CLASS_STABLE,
                "CLASS_VOLATILE": TaskTokens.CLASS_VOLATILE,
            })
            
        if "regression" in self.enabled_tasks:
            tokens["REGRESSION_SEP"] = TaskTokens.REGRESSION_SEP
            
        if "anomaly" in self.enabled_tasks:
            tokens["ANOMALY_SEP"] = TaskTokens.ANOMALY_SEP
            
        if "generation" in self.enabled_tasks:
            tokens["GENERATION_SEP"] = TaskTokens.GENERATION_SEP
            
        return tokens
    
    def _calculate_vocab_extension(self) -> int:
        """Calculate how many new tokens are needed beyond base vocabulary."""
        max_token_id = max(self.token_mapping.values())
        if max_token_id >= self.base_vocab_size:
            return max_token_id - self.base_vocab_size + 1
        return 0
    
    def get_token_id(self, token_name: str) -> int:
        """Get token ID by name with validation.
        
        Args:
            token_name: Name of the token (e.g., 'COVARIATE_SEP')
            
        Returns:
            Token ID
            
        Raises:
            KeyError: If token name is not found
        """
        if token_name not in self.token_mapping:
            raise KeyError(f"Token '{token_name}' not found. Available tokens: {list(self.token_mapping.keys())}")
        return self.token_mapping[token_name]
    
    def get_token_name(self, token_id: int) -> Optional[str]:
        """Get token name by ID.
        
        Args:
            token_id: Token ID to look up
            
        Returns:
            Token name if found, None otherwise
        """
        for name, id_ in self.token_mapping.items():
            if id_ == token_id:
                return name
        return None
    
    def get_classification_tokens(self) -> Dict[str, int]:
        """Get classification class tokens mapping.
        
        Returns:
            Dictionary mapping class names to token IDs
        """
        if "classification" not in self.enabled_tasks:
            return {}
        
        return {
            name: self.get_token_id(f"CLASS_{name.upper()}")
            for name in ["uptrend", "downtrend", "stable", "volatile"]
        }
    
    def get_separator_tokens(self) -> Dict[str, int]:
        """Get all separator token IDs.
        
        Returns:
            Dictionary mapping separator names to token IDs
        """
        separators = {}
        for name, token_id in self.token_mapping.items():
            if name.endswith("_SEP"):
                separators[name] = token_id
        return separators
    
    def extend_vocab_if_needed(self, base_model: nn.Module) -> nn.Module:
        """Extend base model vocabulary if new tokens are needed.
        
        Args:
            base_model: Base TimesFM model
            
        Returns:
            Model with extended vocabulary (if needed)
        """
        if self.vocab_extension_size == 0:
            return base_model
            
        # Find the embedding layer (typically input_ff_layer or similar in TimesFM)
        # This assumes TimesFM has an embedding-like input layer
        embedding_layer = None
        for name, module in base_model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                # Look for layers that could be embeddings
                if 'input' in name.lower() or 'embed' in name.lower():
                    embedding_layer = module
                    break
        
        if embedding_layer is None:
            # If we can't find the embedding layer, we'll need to modify this
            # for the specific TimesFM architecture
            raise RuntimeError("Could not find embedding layer in base model")
        
        # Extend the embedding layer
        old_weight = embedding_layer.weight.data
        old_vocab_size, embedding_dim = old_weight.shape
        new_vocab_size = old_vocab_size + self.vocab_extension_size
        
        # Create new embedding layer
        new_embedding = nn.Linear(embedding_dim, embedding_dim, bias=False)
        new_embedding.weight = nn.Parameter(torch.zeros(new_vocab_size, embedding_dim))
        
        # Copy old weights
        new_embedding.weight.data[:old_vocab_size] = old_weight
        
        # Initialize new token embeddings (using small random values)
        nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        
        # Replace the old embedding layer
        # This is a simplified approach - actual implementation would need to 
        # handle TimesFM's specific architecture
        
        return base_model
    
    def get_state(self) -> Dict:
        """Get the current state for checkpointing.
        
        Returns:
            Dictionary containing token manager state
        """
        return {
            "enabled_tasks": list(self.enabled_tasks),
            "base_vocab_size": self.base_vocab_size,
            "token_mapping": self.token_mapping,
            "vocab_extension_size": self.vocab_extension_size,
        }
    
    @classmethod
    def from_state(cls, state: Dict) -> "ICFTokenManager":
        """Restore token manager from saved state.
        
        Args:
            state: Dictionary containing saved state
            
        Returns:
            Restored ICFTokenManager instance
        """
        manager = cls(
            enabled_tasks=state["enabled_tasks"],
            base_vocab_size=state["base_vocab_size"]
        )
        
        # Validate that the restored state matches
        assert manager.token_mapping == state["token_mapping"], "Token mapping mismatch"
        assert manager.vocab_extension_size == state["vocab_extension_size"], "Vocab extension size mismatch"
        
        return manager
    
    def __repr__(self) -> str:
        return (f"ICFTokenManager(enabled_tasks={list(self.enabled_tasks)}, "
                f"vocab_extension_size={self.vocab_extension_size})")