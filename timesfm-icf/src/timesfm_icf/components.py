"""
ICF-specific components for TimesFM-ICF.

This module provides the learnable components that are specific to the ICF
system, including separator token embeddings and task-specific heads.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICFComponents(nn.Module):
    """ICF-specific learnable components.
    
    Contains all the trainable parameters that are specific to the ICF system,
    separate from the base TimesFM model parameters.
    """
    
    def __init__(
        self,
        base_hidden_size: int = 1280,  # TimesFM hidden size
        num_separator_tokens: int = 10,
        num_classification_classes: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialize ICF components.
        
        Args:
            base_hidden_size: Hidden size of base TimesFM model (1280)
            num_separator_tokens: Number of separator tokens to support
            num_classification_classes: Number of classification classes
            dropout_rate: Dropout rate for task heads
        """
        super().__init__()
        
        self.base_hidden_size = base_hidden_size
        self.num_separator_tokens = num_separator_tokens
        self.num_classification_classes = num_classification_classes
        
        # Learnable separator token embeddings
        # These replace hardcoded separator tokens with learnable representations
        self.separator_embeddings = nn.Embedding(
            num_separator_tokens, 
            base_hidden_size
        )
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            hidden_size=base_hidden_size,
            num_classes=num_classification_classes,
            dropout_rate=dropout_rate
        )
        
        # Future task heads can be added here
        # self.regression_head = RegressionHead(...)
        # self.anomaly_head = AnomalyDetectionHead(...)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize component parameters."""
        # Initialize separator embeddings with small random values
        nn.init.normal_(self.separator_embeddings.weight, mean=0.0, std=0.02)
    
    def get_separator_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for separator token.
        
        Args:
            token_id: Separator token ID
            
        Returns:
            Embedding tensor of shape [hidden_size]
        """
        # Map token_id to embedding index (this would need proper mapping)
        # For now, use a simple modulo to stay within bounds
        embedding_idx = token_id % self.num_separator_tokens
        return self.separator_embeddings(torch.tensor(embedding_idx))
    
    def forward_classification(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification task.
        
        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_size]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        return self.classification_head(hidden_states)
    
    def get_task_head(self, task_name: str) -> Optional[nn.Module]:
        """Get task-specific head by name.
        
        Args:
            task_name: Name of the task ('classification', 'regression', etc.)
            
        Returns:
            Task head module if available, None otherwise
        """
        task_heads = {
            'classification': self.classification_head,
            # Add more task heads as they are implemented
        }
        return task_heads.get(task_name)


class ClassificationHead(nn.Module):
    """Classification head for time series classification tasks."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_classes: int,
        dropout_rate: float = 0.1,
        pooling_strategy: str = 'cls_token'
    ):
        """Initialize classification head.
        
        Args:
            hidden_size: Input hidden size
            num_classes: Number of classification classes
            dropout_rate: Dropout rate
            pooling_strategy: How to pool sequence representations ('cls_token', 'mean', 'max')
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification head weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for classification.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Pool the sequence representation
        pooled = self._pool_sequence(hidden_states, attention_mask)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Get classification logits
        logits = self.classifier(pooled)
        
        return logits
    
    def _pool_sequence(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence representation for classification.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        if self.pooling_strategy == 'cls_token':
            # Use the first token (CLS-like behavior)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling over non-padded positions
            if attention_mask is not None:
                # Mask out padded positions
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence
            if attention_mask is not None:
                # Set padded positions to very negative values before max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(~mask_expanded, -1e9)
            
            return torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")


class SeparatorTokenEmbedding(nn.Module):
    """Specialized embedding layer for separator tokens.
    
    This provides more sophisticated handling of separator tokens with
    position-aware and context-aware embeddings.
    """
    
    def __init__(
        self, 
        num_separators: int, 
        hidden_size: int,
        use_position_embedding: bool = True,
        max_position: int = 512
    ):
        """Initialize separator token embedding.
        
        Args:
            num_separators: Number of different separator types
            hidden_size: Embedding dimension
            use_position_embedding: Whether to add positional information
            max_position: Maximum position for positional embeddings
        """
        super().__init__()
        
        self.num_separators = num_separators
        self.hidden_size = hidden_size
        self.use_position_embedding = use_position_embedding
        
        # Base separator embeddings
        self.separator_embeddings = nn.Embedding(num_separators, hidden_size)
        
        # Optional positional embeddings for separators
        if use_position_embedding:
            self.position_embeddings = nn.Embedding(max_position, hidden_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.separator_embeddings.weight, mean=0.0, std=0.02)
        if self.use_position_embedding:
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        separator_ids: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for separator embeddings.
        
        Args:
            separator_ids: Separator token IDs [batch_size, seq_len] or [seq_len]
            positions: Position indices [batch_size, seq_len] or [seq_len]
            
        Returns:
            Separator embeddings [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
        """
        # Get base separator embeddings
        embeddings = self.separator_embeddings(separator_ids)
        
        # Add positional embeddings if enabled and positions provided
        if self.use_position_embedding and positions is not None:
            pos_embeddings = self.position_embeddings(positions)
            embeddings = embeddings + pos_embeddings
        
        return embeddings


class TaskSpecificLoss(nn.Module):
    """Unified loss computation for multiple tasks."""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        """Initialize task-specific loss.
        
        Args:
            task_weights: Weights for different tasks (default: equal weights)
        """
        super().__init__()
        
        self.task_weights = task_weights or {'forecasting': 1.0, 'classification': 1.0}
        
        # Task-specific loss functions
        self.forecasting_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        task_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute task-specific losses.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            task_mask: Optional masks for each task
            
        Returns:
            Dictionary of losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Forecasting loss
        if 'forecasting' in predictions and 'forecasting' in targets:
            pred_forecast = predictions['forecasting']
            target_forecast = targets['forecasting']
            
            if task_mask and 'forecasting' in task_mask:
                # Apply masking for forecasting
                mask = task_mask['forecasting']
                pred_forecast = pred_forecast[mask]
                target_forecast = target_forecast[mask]
            
            forecast_loss = self.forecasting_loss(pred_forecast, target_forecast)
            losses['forecasting'] = forecast_loss
            total_loss += self.task_weights['forecasting'] * forecast_loss
        
        # Classification loss  
        if 'classification' in predictions and 'classification' in targets:
            pred_class = predictions['classification']
            target_class = targets['classification']
            
            if task_mask and 'classification' in task_mask:
                # Apply masking for classification
                mask = task_mask['classification']
                pred_class = pred_class[mask]
                target_class = target_class[mask]
            
            class_loss = self.classification_loss(pred_class, target_class)
            losses['classification'] = class_loss
            total_loss += self.task_weights['classification'] * class_loss
        
        losses['total'] = total_loss
        return losses