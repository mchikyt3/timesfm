"""
Sequence builder for TimesFM-ICF with pre-allocated token slots.

This module provides the core sequence building functionality that creates
fixed-size input sequences for torch.compile compatibility while maintaining
flexibility through attention masking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .token_manager import ICFTokenManager


@dataclass
class ICFExample:
    """Single example in ICF format.
    
    Represents one time series example with optional covariates and labels
    for different tasks.
    """
    time_series: Union[List[float], np.ndarray]
    covariates: Optional[Union[List[float], np.ndarray]] = None
    class_label: Optional[Union[int, str]] = None
    future_values: Optional[Union[List[float], np.ndarray]] = None  # For forecasting
    
    def __post_init__(self):
        """Convert inputs to numpy arrays for consistency."""
        if isinstance(self.time_series, list):
            self.time_series = np.array(self.time_series, dtype=np.float32)
        if self.covariates is not None and isinstance(self.covariates, list):
            self.covariates = np.array(self.covariates, dtype=np.float32)
        if self.future_values is not None and isinstance(self.future_values, list):
            self.future_values = np.array(self.future_values, dtype=np.float32)


class CompiledSequenceBuilder:
    """Fixed-size sequence builder for torch.compile compatibility.
    
    Creates pre-allocated token templates that maintain fixed input shapes
    while allowing flexible content through attention masking.
    """
    
    # Fixed template configuration for torch.compile compatibility
    DEFAULT_TEMPLATE_CONFIG = {
        'max_examples': 4,                    # Fixed number of example slots
        'max_covariates_per_example': 3,      # Fixed covariate slots per example  
        'max_ts_tokens_per_example': 10,      # Fixed time series slots per example
        'max_class_tokens_per_example': 1,    # Fixed classification slots per example
        'total_sequence_length': 64,          # Fixed total sequence length
    }
    
    def __init__(
        self,
        token_manager: ICFTokenManager,
        template_config: Optional[Dict] = None,
        tokenizer_fn: Optional[callable] = None
    ):
        """Initialize sequence builder.
        
        Args:
            token_manager: ICF token manager for token ID mappings
            template_config: Template configuration override
            tokenizer_fn: Function to tokenize time series data (default: simple binning)
        """
        self.token_manager = token_manager
        self.template_config = template_config or self.DEFAULT_TEMPLATE_CONFIG.copy()
        self.tokenizer_fn = tokenizer_fn or self._default_tokenizer
        
        # Pre-calculate template structure for efficiency
        self._template_structure = self._create_template_structure()
        
    def _default_tokenizer(self, values: np.ndarray, max_tokens: int) -> List[int]:
        """Default tokenization: simple binning approach.
        
        Args:
            values: Array of values to tokenize
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            List of token IDs
        """
        if len(values) == 0:
            return []
        
        # Simple binning strategy: map values to token range
        # This is a placeholder - real implementation would use TimesFM's tokenizer
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            # Handle constant sequences
            token_id = 1000  # Use a default token ID
            return [token_id] * min(len(values), max_tokens)
        
        # Map values to token range [1000, 49999] (avoiding special tokens)
        token_range = (1000, 49999)
        normalized = (values - min_val) / (max_val - min_val)
        token_ids = (normalized * (token_range[1] - token_range[0]) + token_range[0]).astype(int)
        
        # Truncate or pad to max_tokens
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
        
        return token_ids.tolist()
    
    def _create_template_structure(self) -> List[str]:
        """Create the fixed template structure for sequences.
        
        Returns:
            List of slot types defining the template structure
        """
        structure = []
        
        for example_idx in range(self.template_config['max_examples']):
            # Covariate slots
            for cov_idx in range(self.template_config['max_covariates_per_example']):
                structure.append(f'cov_{example_idx}_{cov_idx}')
            
            # Covariate separator
            structure.append(f'cov_sep_{example_idx}')
            
            # Time series slots
            for ts_idx in range(self.template_config['max_ts_tokens_per_example']):
                structure.append(f'ts_{example_idx}_{ts_idx}')
            
            # Classification slots (if classification enabled)
            if 'classification' in self.token_manager.enabled_tasks:
                for cls_idx in range(self.template_config['max_class_tokens_per_example']):
                    structure.append(f'class_{example_idx}_{cls_idx}')
                structure.append(f'class_sep_{example_idx}')
            
            # Example separator (except for last example)
            if example_idx < self.template_config['max_examples'] - 1:
                structure.append(f'example_sep_{example_idx}')
        
        # Pad to total sequence length
        while len(structure) < self.template_config['total_sequence_length']:
            structure.append('padding')
            
        # Truncate if necessary
        structure = structure[:self.template_config['total_sequence_length']]
        
        return structure
    
    def build_sequence(
        self, 
        examples: List[ICFExample],
        task_type: str = 'forecasting'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-size sequence with attention masking.
        
        Args:
            examples: List of ICF examples (will be padded/truncated to max_examples)
            task_type: Type of task ('forecasting', 'classification', etc.)
            
        Returns:
            Tuple of (sequence_tensor, attention_mask_tensor)
            Both tensors have shape [total_sequence_length]
        """
        # Initialize sequence with padding tokens
        sequence = [self.token_manager.get_token_id('PADDING')] * self.template_config['total_sequence_length']
        attention_mask = [0] * self.template_config['total_sequence_length']  # 0 = ignore
        
        # Process examples (truncate or pad to max_examples)
        padded_examples = self._pad_examples(examples)
        
        # Fill template according to structure
        pos = 0
        for example_idx in range(self.template_config['max_examples']):
            example = padded_examples[example_idx] if example_idx < len(padded_examples) else None
            pos = self._fill_example_slots(sequence, attention_mask, pos, example, example_idx, task_type)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.bool)
    
    def _pad_examples(self, examples: List[ICFExample]) -> List[Optional[ICFExample]]:
        """Pad or truncate examples list to max_examples.
        
        Args:
            examples: Original examples list
            
        Returns:
            List padded/truncated to max_examples (None for empty slots)
        """
        max_examples = self.template_config['max_examples']
        
        if len(examples) >= max_examples:
            return examples[:max_examples]
        else:
            # Pad with None for empty slots
            return examples + [None] * (max_examples - len(examples))
    
    def _fill_example_slots(
        self,
        sequence: List[int],
        attention_mask: List[int], 
        start_pos: int,
        example: Optional[ICFExample],
        example_idx: int,
        task_type: str
    ) -> int:
        """Fill template slots for one example.
        
        Args:
            sequence: Sequence being built
            attention_mask: Attention mask being built
            start_pos: Starting position in sequence
            example: Example to process (None for empty slot)
            example_idx: Index of this example
            task_type: Task type being processed
            
        Returns:
            Next position in sequence
        """
        pos = start_pos
        
        # Fill covariate slots
        cov_tokens = []
        if example is not None and example.covariates is not None:
            cov_tokens = self._tokenize_covariates(example.covariates)
        
        max_cov_slots = self.template_config['max_covariates_per_example']
        for i in range(max_cov_slots):
            if i < len(cov_tokens):
                sequence[pos] = cov_tokens[i]
                attention_mask[pos] = 1  # Attend to real covariate
            # else: keep padding token with mask=0
            pos += 1
        
        # Covariate separator (always present for structure consistency)
        sequence[pos] = self.token_manager.get_token_id('COVARIATE_SEP')
        attention_mask[pos] = 1 if example is not None else 0
        pos += 1
        
        # Fill time series slots
        ts_tokens = []
        if example is not None:
            ts_tokens = self.tokenizer_fn(
                example.time_series, 
                self.template_config['max_ts_tokens_per_example']
            )
        
        max_ts_slots = self.template_config['max_ts_tokens_per_example']
        for i in range(max_ts_slots):
            if i < len(ts_tokens):
                sequence[pos] = ts_tokens[i]
                attention_mask[pos] = 1  # Attend to real time series token
            # else: keep padding token with mask=0
            pos += 1
        
        # Fill classification slots (if classification enabled)
        if 'classification' in self.token_manager.enabled_tasks:
            class_tokens = []
            if example is not None and example.class_label is not None and task_type == 'classification':
                class_token_id = self._get_class_token_id(example.class_label)
                if class_token_id is not None:
                    class_tokens = [class_token_id]
            
            max_class_slots = self.template_config['max_class_tokens_per_example']
            for i in range(max_class_slots):
                if i < len(class_tokens):
                    sequence[pos] = class_tokens[i]
                    attention_mask[pos] = 1  # Attend to real class token
                # else: keep padding token with mask=0
                pos += 1
            
            # Classification separator
            sequence[pos] = self.token_manager.get_token_id('CLASS_SEP')
            attention_mask[pos] = 1 if example is not None and task_type == 'classification' else 0
            pos += 1
        
        # Example separator (except for last example)
        if example_idx < self.template_config['max_examples'] - 1:
            sequence[pos] = self.token_manager.get_token_id('EXAMPLE_SEP')
            attention_mask[pos] = 1 if example is not None else 0
            pos += 1
        
        return pos
    
    def _tokenize_covariates(self, covariates: np.ndarray) -> List[int]:
        """Tokenize covariate values.
        
        Args:
            covariates: Covariate values array
            
        Returns:
            List of covariate token IDs
        """
        max_cov_tokens = self.template_config['max_covariates_per_example']
        return self.tokenizer_fn(covariates, max_cov_tokens)
    
    def _get_class_token_id(self, class_label: Union[int, str]) -> Optional[int]:
        """Get token ID for class label.
        
        Args:
            class_label: Class label (string name or integer ID)
            
        Returns:
            Token ID if valid, None otherwise
        """
        if isinstance(class_label, str):
            # Map string labels to token IDs
            class_tokens = self.token_manager.get_classification_tokens()
            return class_tokens.get(class_label.lower())
        elif isinstance(class_label, int):
            # Direct token ID (validate it's a valid class token)
            class_tokens = self.token_manager.get_classification_tokens()
            if class_label in class_tokens.values():
                return class_label
        
        return None
    
    def create_dummy_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy batch for torch.compile tracing.
        
        Args:
            batch_size: Size of dummy batch
            
        Returns:
            Tuple of (dummy_sequences, dummy_masks) with shape [batch_size, seq_len]
        """
        dummy_sequence = torch.zeros(
            (batch_size, self.template_config['total_sequence_length']), 
            dtype=torch.long
        )
        dummy_mask = torch.zeros(
            (batch_size, self.template_config['total_sequence_length']), 
            dtype=torch.bool
        )
        return dummy_sequence, dummy_mask
    
    def get_template_info(self) -> Dict:
        """Get information about the current template configuration.
        
        Returns:
            Dictionary with template configuration and structure info
        """
        return {
            'template_config': self.template_config.copy(),
            'template_structure': self._template_structure.copy(),
            'enabled_tasks': list(self.token_manager.enabled_tasks),
            'total_slots': len(self._template_structure),
        }
    
    def __repr__(self) -> str:
        return (f"CompiledSequenceBuilder("
                f"max_examples={self.template_config['max_examples']}, "
                f"seq_len={self.template_config['total_sequence_length']}, "
                f"tasks={list(self.token_manager.enabled_tasks)})")