"""
Main TimesFM-ICF model that integrates all components.

This module provides the main TimesFMICF class that wraps the base TimesFM
model and adds ICF capabilities through learnable separator tokens.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Import TimesFM components (these would be actual imports in production)
try:
  from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

  from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
except ImportError:
  # Fallback for development environment
  TimesFm = None
  TimesFmCheckpoint = None
  TimesFmHparams = None
  PatchedTimeSeriesDecoder = None

from timesfm_icf.components import ICFComponents
from timesfm_icf.model_config import ICFConfig
from timesfm_icf.sequence_builder import CompiledSequenceBuilder, ICFExample
from timesfm_icf.token_manager import ICFTokenManager


class TimesFMICF(nn.Module):
  """Main ICF wrapper around TimesFM model.

  Provides In-Context Fine-Tuning capabilities by extending TimesFM with
  learnable separator tokens and task-specific heads.
  """

  def __init__(self, config: Optional[ICFConfig] = None):
    """Initialize TimesFM-ICF model.

    Args:
        config: ICF configuration (uses default if not provided)
    """
    super().__init__()

    # Use default config if not provided
    self.config = config or ICFConfig()

    # Core components (initialized later)
    self.base_model: Optional[PatchedTimeSeriesDecoder] = None
    self.token_manager: Optional[ICFTokenManager] = None
    self.sequence_builder: Optional[CompiledSequenceBuilder] = None
    self.icf_components: Optional[ICFComponents] = None

    # Compilation state
    self.is_compiled = False
    self.compiled_forward = None

    # Device management
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    self.logger = logging.getLogger(__name__)

  def load_timesfm_checkpoint(
    self, checkpoint: Union[TimesFmCheckpoint, str, Dict[str, Any]]
  ):
    """Load base TimesFM model using existing loading system.

    Args:
        checkpoint: TimesFM checkpoint (object, path, or config dict)
    """
    if TimesFm is None:
      raise ImportError("TimesFM package not available. Please install timesfm first.")

    self.logger.info("Loading base TimesFM model...")

    if isinstance(checkpoint, str):
      # Treat as Hugging Face repo ID
      checkpoint = TimesFmCheckpoint(huggingface_repo_id=checkpoint)
    elif isinstance(checkpoint, dict):
      # Create checkpoint from dictionary
      checkpoint = TimesFmCheckpoint(**checkpoint)

    # Initialize base TimesFM model
    # Note: This is a simplified version - real implementation would need
    # to handle TimesFM's specific initialization pattern
    base_timesfm = TimesFm(hparams=TimesFmHparams(), backend="torch")
    base_timesfm.load_from_checkpoint(checkpoint)

    # Extract the PyTorch model
    self.base_model = base_timesfm._model
    self.base_model.to(self.device)

    self.logger.info("Base TimesFM model loaded successfully")

  def initialize_icf_components(self, enabled_tasks: List[str]):
    """Initialize ICF-specific components after base model loading.

    Args:
        enabled_tasks: List of tasks to enable (e.g., ['forecasting', 'classification'])
    """
    if self.base_model is None:
      raise RuntimeError(
        "Must load TimesFM checkpoint before initializing ICF components"
      )

    self.logger.info(f"Initializing ICF components for tasks: {enabled_tasks}")

    # Initialize token manager
    self.token_manager = ICFTokenManager(
      enabled_tasks=enabled_tasks, base_vocab_size=self.config.base_vocab_size
    )

    # Extend vocabulary if needed
    if self.token_manager.vocab_extension_size > 0:
      self.logger.info(
        f"Extending vocabulary by {self.token_manager.vocab_extension_size} tokens"
      )
      self.base_model = self.token_manager.extend_vocab_if_needed(self.base_model)

    # Initialize sequence builder
    self.sequence_builder = CompiledSequenceBuilder(
      token_manager=self.token_manager, template_config=self.config.template_config
    )

    # Initialize ICF components
    self.icf_components = ICFComponents(
      base_hidden_size=self.config.hidden_size,
      num_separator_tokens=len(self.token_manager.get_separator_tokens()),
      num_classification_classes=self.config.num_classification_classes,
      dropout_rate=self.config.dropout_rate,
    )
    self.icf_components.to(self.device)

    self.logger.info("ICF components initialized successfully")

  def compile_model(self, mode: str = "default"):
    """Apply torch.compile with fixed input shapes.

    Args:
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
    """
    if self.base_model is None or self.sequence_builder is None:
      raise RuntimeError("Must initialize all components before compilation")

    self.logger.info(f"Compiling model with mode: {mode}")

    # Create dummy inputs for tracing
    dummy_sequences, dummy_masks = self.sequence_builder.create_dummy_batch(
      batch_size=1
    )
    dummy_sequences = dummy_sequences.to(self.device)
    dummy_masks = dummy_masks.to(self.device)

    # Compile the forward pass
    self.compiled_forward = torch.compile(
      self._forward_wrapper,
      mode=mode,
      fullgraph=True,  # Ensure no graph breaks
    )

    # Warm up compilation with dummy inputs
    try:
      with torch.no_grad():
        _ = self.compiled_forward(dummy_sequences, dummy_masks)
      self.is_compiled = True
      self.logger.info("Model compilation successful")
    except Exception as e:
      self.logger.warning(f"Model compilation failed: {e}")
      self.compiled_forward = None
      self.is_compiled = False

  def _forward_wrapper(
    self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    """Wrapper for compiled forward pass with fixed signature.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] (1=attend, 0=ignore)

    Returns:
        Model outputs [batch_size, seq_len, hidden_size]
    """
    # Convert attention mask format for TimesFM (1=pad, 0=real)
    timesfm_padding = 1 - attention_mask.float()

    # Create frequency tensor (simplified - would need proper frequency handling)
    batch_size = input_ids.shape[0]
    freq_tensor = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    # Forward pass through base TimesFM model
    return self.base_model(
      input_ts=input_ids.float(),  # TimesFM expects float input
      input_padding=timesfm_padding,
      freq=freq_tensor,
    )

  def forecast(
    self, examples: List[ICFExample], horizon: int = 24, task_type: str = "forecasting"
  ) -> torch.Tensor:
    """Perform forecasting on examples.

    Args:
        examples: List of ICF examples
        horizon: Forecast horizon
        task_type: Task type ('forecasting')

    Returns:
        Forecast predictions [batch_size, horizon]
    """
    self.eval()

    with torch.no_grad():
      # Build input sequence
      input_ids, attention_mask = self.sequence_builder.build_sequence(
        examples, task_type
      )
      input_ids = input_ids.unsqueeze(0).to(self.device)  # Add batch dimension
      attention_mask = attention_mask.unsqueeze(0).to(self.device)

      # Forward pass
      if self.is_compiled and self.compiled_forward is not None:
        outputs = self.compiled_forward(input_ids, attention_mask)
      else:
        outputs = self._forward_wrapper(input_ids, attention_mask)

      # Extract forecasting predictions (this would need proper implementation
      # based on TimesFM's output format)
      return outputs  # Placeholder

  def classify(
    self, examples: List[ICFExample], return_probabilities: bool = False
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Perform classification on examples.

    Args:
        examples: List of ICF examples
        return_probabilities: Whether to return class probabilities

    Returns:
        Class predictions [batch_size] or (predictions, probabilities)
    """
    if "classification" not in self.token_manager.enabled_tasks:
      raise ValueError("Classification task not enabled")

    self.eval()

    with torch.no_grad():
      # Build input sequence
      input_ids, attention_mask = self.sequence_builder.build_sequence(
        examples, "classification"
      )
      input_ids = input_ids.unsqueeze(0).to(self.device)
      attention_mask = attention_mask.unsqueeze(0).to(self.device)

      # Forward pass through base model
      if self.is_compiled and self.compiled_forward is not None:
        hidden_states = self.compiled_forward(input_ids, attention_mask)
      else:
        hidden_states = self._forward_wrapper(input_ids, attention_mask)

      # Classification head forward pass
      logits = self.icf_components.forward_classification(hidden_states)

      # Get predictions
      probabilities = torch.softmax(logits, dim=-1)
      predictions = torch.argmax(logits, dim=-1)

      if return_probabilities:
        return predictions, probabilities
      else:
        return predictions

  def save_icf_checkpoint(self, path: str, metadata: Optional[Dict] = None):
    """Save ICF-specific components and metadata.

    Args:
        path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    if self.icf_components is None or self.token_manager is None:
      raise RuntimeError("ICF components not initialized")

    checkpoint_data = {
      # ICF component states
      "icf_components_state_dict": self.icf_components.state_dict(),
      # Configuration and token manager state
      "config": self.config.__dict__,
      "token_manager_state": self.token_manager.get_state(),
      "sequence_builder_config": self.sequence_builder.get_template_info(),
      # Model state (if base model was fine-tuned)
      "base_model_state_dict": self.base_model.state_dict()
      if self.base_model
      else None,
      # Metadata
      "metadata": metadata or {},
      "version": "1.0.0",
    }

    torch.save(checkpoint_data, path)
    self.logger.info(f"ICF checkpoint saved to {path}")

  def load_icf_checkpoint(self, path: str):
    """Load ICF checkpoint on top of base TimesFM.

    Args:
        path: Path to ICF checkpoint
    """
    if self.base_model is None:
      raise RuntimeError("Must load base TimesFM checkpoint first")

    checkpoint_data = torch.load(path, map_location=self.device, weights_only=True)

    # Restore token manager
    self.token_manager = ICFTokenManager.from_state(
      checkpoint_data["token_manager_state"]
    )

    # Restore sequence builder
    self.sequence_builder = CompiledSequenceBuilder(
      token_manager=self.token_manager,
      template_config=checkpoint_data["sequence_builder_config"]["template_config"],
    )

    # Restore ICF components
    self.icf_components = ICFComponents(
      base_hidden_size=checkpoint_data["config"]["hidden_size"],
      num_separator_tokens=len(self.token_manager.get_separator_tokens()),
      num_classification_classes=checkpoint_data["config"][
        "num_classification_classes"
      ],
      dropout_rate=checkpoint_data["config"]["dropout_rate"],
    )
    self.icf_components.load_state_dict(checkpoint_data["icf_components_state_dict"])
    self.icf_components.to(self.device)

    # Load base model state if it was fine-tuned
    if checkpoint_data["base_model_state_dict"] is not None:
      self.base_model.load_state_dict(checkpoint_data["base_model_state_dict"])

    self.logger.info(f"ICF checkpoint loaded from {path}")

  def get_model_info(self) -> Dict[str, Any]:
    """Get information about the current model state.

    Returns:
        Dictionary with model information
    """
    info = {
      "base_model_loaded": self.base_model is not None,
      "icf_components_initialized": self.icf_components is not None,
      "is_compiled": self.is_compiled,
      "device": str(self.device),
      "config": self.config.__dict__ if self.config else None,
    }

    if self.token_manager:
      info["enabled_tasks"] = list(self.token_manager.enabled_tasks)
      info["vocab_extension_size"] = self.token_manager.vocab_extension_size

    if self.sequence_builder:
      info["template_info"] = self.sequence_builder.get_template_info()

    return info

  def __repr__(self) -> str:
    status = (
      "initialized" if self.base_model and self.icf_components else "uninitialized"
    )
    compiled_status = "compiled" if self.is_compiled else "not compiled"
    return f"TimesFMICF(status={status}, {compiled_status}, device={self.device})"
