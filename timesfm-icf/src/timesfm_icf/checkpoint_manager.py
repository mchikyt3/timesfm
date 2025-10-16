"""
Checkpoint management for TimesFM-ICF models.

This module provides comprehensive checkpoint save/load functionality for ICF models,
including base TimesFM checkpoints and ICF-specific components.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

# TimesFM imports
try:
  from timesfm import TimesFmCheckpoint
except ImportError:
  TimesFmCheckpoint = None

from timesfm_icf.model_config import ICFConfig


@dataclass
class ICFCheckpoint:
  """ICF model checkpoint information.

  Contains all information needed to restore an ICF model, including
  both base TimesFM checkpoint info and ICF-specific components.
  """

  # Base TimesFM checkpoint information
  base_timesfm_checkpoint: Optional[Union[TimesFmCheckpoint, Dict[str, Any]]] = None
  """Base TimesFM checkpoint (object or dict)"""

  # ICF-specific checkpoint information
  icf_checkpoint_path: Optional[str] = None
  """Path to ICF components checkpoint"""

  enabled_tasks: List[str] = field(default_factory=list)
  """List of enabled tasks"""

  token_mapping: Dict[str, int] = field(default_factory=dict)
  """Token name to ID mapping"""

  # Model configuration
  icf_config: Optional[ICFConfig] = None
  """ICF model configuration"""

  # Training metadata
  training_metadata: Dict[str, Any] = field(default_factory=dict)
  """Training history and metrics"""

  # Checkpoint metadata
  version: str = "0.1.0"
  """Checkpoint format version"""

  created_at: Optional[str] = None
  """Checkpoint creation timestamp"""

  description: Optional[str] = None
  """Human-readable description of the checkpoint"""

  def __post_init__(self):
    """Validate checkpoint configuration."""
    if self.base_timesfm_checkpoint is None and self.icf_checkpoint_path is None:
      raise ValueError(
        "Must specify either base_timesfm_checkpoint or icf_checkpoint_path"
      )

    # Convert TimesFmCheckpoint to dict for serialization if needed
    if TimesFmCheckpoint and isinstance(
      self.base_timesfm_checkpoint, TimesFmCheckpoint
    ):
      self.base_timesfm_checkpoint = asdict(self.base_timesfm_checkpoint)

  def to_dict(self) -> Dict[str, Any]:
    """Convert checkpoint info to dictionary for serialization."""
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ICFCheckpoint":
    """Create checkpoint info from dictionary."""
    return cls(**data)


class ICFCheckpointManager:
  """Manages saving and loading of ICF models.

  Provides comprehensive checkpoint management including:
  - Full model checkpoints (base + ICF components)
  - ICF-only checkpoints (for pre-trained base models)
  - Checkpoint metadata and versioning
  - Automatic checkpoint organization
  """

  def __init__(self, checkpoint_dir: str = "./checkpoints"):
    """Initialize checkpoint manager.

    Args:
        checkpoint_dir: Directory for storing checkpoints
    """
    self.checkpoint_dir = checkpoint_dir
    self.logger = logging.getLogger(__name__)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create subdirectories
    self.icf_dir = os.path.join(checkpoint_dir, "icf")
    self.full_dir = os.path.join(checkpoint_dir, "full")
    self.metadata_dir = os.path.join(checkpoint_dir, "metadata")

    for dir_path in [self.icf_dir, self.full_dir, self.metadata_dir]:
      os.makedirs(dir_path, exist_ok=True)

  def save_full_checkpoint(
    self,
    model,  # TimesFMICF instance
    checkpoint_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
  ) -> str:
    """Save complete ICF model checkpoint (base + ICF components).

    Args:
        model: TimesFMICF model instance
        checkpoint_name: Name for the checkpoint
        metadata: Additional metadata to save
        description: Human-readable description

    Returns:
        Path to saved checkpoint
    """
    if not model.base_model or not model.icf_components:
      raise RuntimeError("Model must be fully initialized to save full checkpoint")

    self.logger.info(f"Saving full checkpoint: {checkpoint_name}")

    # Prepare checkpoint data
    checkpoint_data = {
      # Model states
      "base_model_state_dict": model.base_model.state_dict(),
      "icf_components_state_dict": model.icf_components.state_dict(),
      # ICF configuration
      "config": asdict(model.config)
      if hasattr(model.config, "__dict__")
      else model.config,
      "token_manager_state": model.token_manager.get_state()
      if model.token_manager
      else None,
      "sequence_builder_config": model.sequence_builder.get_template_info()
      if model.sequence_builder
      else None,
      # Model info
      "model_info": model.get_model_info(),
      "enabled_tasks": list(model.token_manager.enabled_tasks)
      if model.token_manager
      else [],
      # Metadata
      "metadata": metadata or {},
      "description": description,
      "version": "0.1.0",
      "checkpoint_type": "full",
    }

    # Save checkpoint
    checkpoint_path = os.path.join(self.full_dir, f"{checkpoint_name}.pt")
    torch.save(checkpoint_data, checkpoint_path)

    # Save metadata separately for easy inspection
    self._save_checkpoint_metadata(checkpoint_name, checkpoint_data, "full")

    self.logger.info(f"Full checkpoint saved to {checkpoint_path}")
    return checkpoint_path

  def save_icf_checkpoint(
    self,
    model,  # TimesFMICF instance
    checkpoint_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
  ) -> str:
    """Save ICF-only checkpoint (without base model weights).

    Args:
        model: TimesFMICF model instance
        checkpoint_name: Name for the checkpoint
        metadata: Additional metadata to save
        description: Human-readable description

    Returns:
        Path to saved checkpoint
    """
    if not model.icf_components:
      raise RuntimeError("ICF components must be initialized to save ICF checkpoint")

    self.logger.info(f"Saving ICF checkpoint: {checkpoint_name}")

    # Prepare checkpoint data (ICF components only)
    checkpoint_data = {
      # ICF states only
      "icf_components_state_dict": model.icf_components.state_dict(),
      # ICF configuration
      "config": asdict(model.config)
      if hasattr(model.config, "__dict__")
      else model.config,
      "token_manager_state": model.token_manager.get_state()
      if model.token_manager
      else None,
      "sequence_builder_config": model.sequence_builder.get_template_info()
      if model.sequence_builder
      else None,
      # Model info (without base model state)
      "enabled_tasks": list(model.token_manager.enabled_tasks)
      if model.token_manager
      else [],
      # Metadata
      "metadata": metadata or {},
      "description": description,
      "version": "0.1.0",
      "checkpoint_type": "icf_only",
    }

    # Save checkpoint
    checkpoint_path = os.path.join(self.icf_dir, f"{checkpoint_name}.pt")
    torch.save(checkpoint_data, checkpoint_path)

    # Save metadata separately
    self._save_checkpoint_metadata(checkpoint_name, checkpoint_data, "icf_only")

    self.logger.info(f"ICF checkpoint saved to {checkpoint_path}")
    return checkpoint_path

  def load_from_checkpoint(
    self, checkpoint: ICFCheckpoint, device: Optional[str] = None
  ):  # -> TimesFMICF:
    """Load complete ICF model from checkpoint.

    Args:
        checkpoint: ICF checkpoint information
        device: Device to load model on

    Returns:
        Loaded TimesFMICF model
    """
    # Import here to avoid circular imports
    from timesfm_icf.icf_model import TimesFMICF

    self.logger.info("Loading ICF model from checkpoint")

    # Initialize model
    model = TimesFMICF(config=checkpoint.icf_config)

    if device:
      model.device = torch.device(device)

    # Load base TimesFM model if specified
    if checkpoint.base_timesfm_checkpoint:
      if isinstance(checkpoint.base_timesfm_checkpoint, dict):
        # Convert dict back to TimesFmCheckpoint if needed
        if TimesFmCheckpoint:
          base_checkpoint = TimesFmCheckpoint(**checkpoint.base_timesfm_checkpoint)
        else:
          base_checkpoint = checkpoint.base_timesfm_checkpoint
      else:
        base_checkpoint = checkpoint.base_timesfm_checkpoint

      model.load_timesfm_checkpoint(base_checkpoint)

    # Load ICF checkpoint if specified
    if checkpoint.icf_checkpoint_path:
      if os.path.exists(checkpoint.icf_checkpoint_path):
        model.load_icf_checkpoint(checkpoint.icf_checkpoint_path)
      else:
        # Try relative to checkpoint directory
        relative_path = os.path.join(
          self.checkpoint_dir, checkpoint.icf_checkpoint_path
        )
        if os.path.exists(relative_path):
          model.load_icf_checkpoint(relative_path)
        else:
          raise FileNotFoundError(
            f"ICF checkpoint not found: {checkpoint.icf_checkpoint_path}"
          )

    # Initialize ICF components if not already done
    if checkpoint.enabled_tasks and not model.icf_components:
      model.initialize_icf_components(checkpoint.enabled_tasks)

    self.logger.info("ICF model loaded successfully")
    return model

  def load_checkpoint_by_name(
    self,
    checkpoint_name: str,
    checkpoint_type: str = "auto",
    device: Optional[str] = None,
  ):  # -> TimesFMICF:
    """Load checkpoint by name.

    Args:
        checkpoint_name: Name of the checkpoint to load
        checkpoint_type: Type of checkpoint ('full', 'icf_only', 'auto')
        device: Device to load model on

    Returns:
        Loaded TimesFMICF model
    """
    # Import here to avoid circular imports
    from timesfm_icf.icf_model import TimesFMICF

    # Determine checkpoint path
    if checkpoint_type == "auto":
      # Try full checkpoint first, then ICF-only
      full_path = os.path.join(self.full_dir, f"{checkpoint_name}.pt")
      icf_path = os.path.join(self.icf_dir, f"{checkpoint_name}.pt")

      if os.path.exists(full_path):
        checkpoint_path = full_path
        checkpoint_type = "full"
      elif os.path.exists(icf_path):
        checkpoint_path = icf_path
        checkpoint_type = "icf_only"
      else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    elif checkpoint_type == "full":
      checkpoint_path = os.path.join(self.full_dir, f"{checkpoint_name}.pt")
    elif checkpoint_type == "icf_only":
      checkpoint_path = os.path.join(self.icf_dir, f"{checkpoint_name}.pt")
    else:
      raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")

    if not os.path.exists(checkpoint_path):
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint data
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Create model from checkpoint
    config = (
      ICFConfig(**checkpoint_data["config"])
      if "config" in checkpoint_data
      else ICFConfig()
    )
    model = TimesFMICF(config=config)

    if device:
      model.device = torch.device(device)

    # Load model states
    if checkpoint_type == "full" and "base_model_state_dict" in checkpoint_data:
      # For full checkpoints, we need to reconstruct the base model
      # This is simplified - real implementation would need proper TimesFM loading
      self.logger.warning(
        "Full checkpoint loading not fully implemented - loading ICF components only"
      )

    # Load ICF components
    if "icf_components_state_dict" in checkpoint_data:
      # Initialize ICF components first
      enabled_tasks = checkpoint_data.get("enabled_tasks", ["forecasting"])
      model.initialize_icf_components(enabled_tasks)

      # Load ICF component states
      model.icf_components.load_state_dict(checkpoint_data["icf_components_state_dict"])

    self.logger.info(f"Checkpoint loaded: {checkpoint_name} ({checkpoint_type})")
    return model

  def list_checkpoints(self) -> Dict[str, List[str]]:
    """List available checkpoints.

    Returns:
        Dictionary with checkpoint types and their available names
    """
    checkpoints = {
      "full": [],
      "icf_only": [],
    }

    # List full checkpoints
    if os.path.exists(self.full_dir):
      for file in os.listdir(self.full_dir):
        if file.endswith(".pt"):
          checkpoints["full"].append(file[:-3])  # Remove .pt extension

    # List ICF-only checkpoints
    if os.path.exists(self.icf_dir):
      for file in os.listdir(self.icf_dir):
        if file.endswith(".pt"):
          checkpoints["icf_only"].append(file[:-3])  # Remove .pt extension

    return checkpoints

  def get_checkpoint_metadata(
    self, checkpoint_name: str, checkpoint_type: str = "auto"
  ) -> Dict[str, Any]:
    """Get metadata for a specific checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint
        checkpoint_type: Type of checkpoint ('full', 'icf_only', 'auto')

    Returns:
        Checkpoint metadata dictionary
    """
    # Try to find metadata file
    metadata_file = os.path.join(
      self.metadata_dir, f"{checkpoint_name}_{checkpoint_type}.json"
    )

    if checkpoint_type == "auto":
      # Try both types
      for ctype in ["full", "icf_only"]:
        metadata_file = os.path.join(
          self.metadata_dir, f"{checkpoint_name}_{ctype}.json"
        )
        if os.path.exists(metadata_file):
          break

    if os.path.exists(metadata_file):
      with open(metadata_file, "r") as f:
        return json.load(f)
    else:
      return {}

  def _save_checkpoint_metadata(
    self, checkpoint_name: str, checkpoint_data: Dict[str, Any], checkpoint_type: str
  ):
    """Save checkpoint metadata to separate file for easy inspection."""
    metadata = {
      "checkpoint_name": checkpoint_name,
      "checkpoint_type": checkpoint_type,
      "version": checkpoint_data.get("version"),
      "description": checkpoint_data.get("description"),
      "enabled_tasks": checkpoint_data.get("enabled_tasks"),
      "metadata": checkpoint_data.get("metadata"),
      "config_summary": self._extract_config_summary(checkpoint_data.get("config")),
    }

    metadata_file = os.path.join(
      self.metadata_dir, f"{checkpoint_name}_{checkpoint_type}.json"
    )
    with open(metadata_file, "w") as f:
      json.dump(metadata, f, indent=2)

  def _extract_config_summary(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract key configuration parameters for metadata."""
    if not config:
      return {}

    return {
      "enabled_tasks": config.get("enabled_tasks"),
      "max_examples": config.get("max_examples"),
      "total_sequence_length": config.get("total_sequence_length"),
      "num_classification_classes": config.get("num_classification_classes"),
      "enable_torch_compile": config.get("enable_torch_compile"),
    }

  def cleanup_old_checkpoints(self, keep_last_n: int = 5):
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        keep_last_n: Number of recent checkpoints to keep
    """
    for checkpoint_dir in [self.full_dir, self.icf_dir]:
      if not os.path.exists(checkpoint_dir):
        continue

      # Get checkpoint files sorted by modification time
      files = []
      for file in os.listdir(checkpoint_dir):
        if file.endswith(".pt"):
          file_path = os.path.join(checkpoint_dir, file)
          files.append((file_path, os.path.getmtime(file_path)))

      # Sort by modification time (newest first)
      files.sort(key=lambda x: x[1], reverse=True)

      # Remove old files
      for file_path, _ in files[keep_last_n:]:
        os.remove(file_path)
        self.logger.info(f"Removed old checkpoint: {file_path}")

        # Also remove corresponding metadata
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        checkpoint_type = "full" if "full" in checkpoint_dir else "icf_only"
        metadata_file = os.path.join(
          self.metadata_dir, f"{base_name}_{checkpoint_type}.json"
        )
        if os.path.exists(metadata_file):
          os.remove(metadata_file)
