"""
Basic usage example for TimesFM-ICF.

This example demonstrates how to:
1. Load a pretrained TimesFM model
2. Initialize ICF components
3. Perform forecasting with covariates
4. Save and load ICF checkpoints
"""

import numpy as np
import torch

from timesfm_icf import (
  ICFCheckpoint,
  ICFCheckpointManager,
  ICFConfig,
  ICFExample,
  TimesFMICF,
)

# Note: This example assumes TimesFM is installed
# pip install timesfm-icf
# The timesfm package will be installed as a dependency


def main():
  print("🚀 TimesFM-ICF Basic Usage Example")
  print("=" * 50)

  # Step 1: Create ICF configuration
  print("\n1. Setting up ICF configuration...")
  config = ICFConfig(
    enabled_tasks=["forecasting"],
    max_examples=4,
    max_ts_tokens_per_example=10,
    max_covariates_per_example=3,
    total_sequence_length=64,
    enable_torch_compile=True,
  )
  print(f"   ✓ Configuration created: {config.enabled_tasks}")

  # Step 2: Initialize TimesFM-ICF model
  print("\n2. Initializing TimesFM-ICF model...")
  model = TimesFMICF(config=config)
  print(f"   ✓ Model initialized on device: {model.device}")

  # Step 3: Load pretrained TimesFM model
  print("\n3. Loading pretrained TimesFM model...")
  try:
    # Load from Hugging Face (this would be the actual repo when available)
    model.load_timesfm_checkpoint("google/timesfm-1.0-200m")
    print("   ✓ TimesFM base model loaded successfully")
  except Exception as e:
    print(f"   ⚠️  Could not load TimesFM model: {e}")
    print("   📝 This is expected in development environment")
    print("   📝 In production, ensure 'timesfm' package is installed")
    return

  # Step 4: Initialize ICF components
  print("\n4. Initializing ICF components...")
  model.initialize_icf_components(["forecasting"])
  print(
    f"   ✓ ICF components initialized for tasks: {model.token_manager.enabled_tasks}"
  )

  # Step 5: Compile model for optimal performance
  print("\n5. Compiling model with torch.compile...")
  try:
    model.compile_model(mode="max-autotune")
    print(f"   ✓ Model compiled: {model.is_compiled}")
  except Exception as e:
    print(f"   ⚠️  Compilation failed: {e}")
    print("   📝 Model will run without compilation")

  # Step 6: Prepare example data
  print("\n6. Preparing example time series data...")

  # Create sample time series with covariates
  examples = [
    ICFExample(
      time_series=np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 2.0]),
      covariates=np.array([0.1, 0.2, 0.3]),  # Temperature, humidity, etc.
    ),
    ICFExample(
      time_series=np.array([0.5, 0.7, 0.9, 1.1, 1.0, 0.8, 0.6]),
      covariates=np.array([0.2, 0.1]),  # Fewer covariates
    ),
    ICFExample(
      time_series=np.array([2.0, 2.5, 3.0, 3.2, 3.1, 2.9]),
      covariates=np.array([0.3, 0.4, 0.5]),  # Different pattern
    ),
  ]

  print(f"   ✓ Created {len(examples)} time series examples")
  print(
    f"   📊 Example 1: TS length={len(examples[0].time_series)}, Covariates={len(examples[0].covariates)}"
  )

  # Step 7: Build sequence and show structure
  print("\n7. Building fixed-size sequence...")
  sequence, attention_mask = model.sequence_builder.build_sequence(
    examples, "forecasting"
  )

  print(f"   ✓ Sequence shape: {sequence.shape}")
  print(f"   ✓ Attention mask shape: {attention_mask.shape}")
  print(
    f"   📊 Real tokens: {attention_mask.sum().item()}/{len(attention_mask)} (padding: {(~attention_mask).sum().item()})"
  )

  # Show sequence structure
  template_info = model.sequence_builder.get_template_info()
  print(f"   📋 Template: {template_info['template_config']}")

  # Step 8: Perform forecasting
  print("\n8. Performing forecasting...")
  try:
    with torch.no_grad():
      predictions = model.forecast(examples, horizon=24)
    print("   ✓ Forecasting completed")
    print(f"   📈 Predictions shape: {predictions.shape}")
    print(f"   📊 Sample predictions: {predictions.flatten()[:5].tolist()}")
  except Exception as e:
    print(f"   ⚠️  Forecasting failed: {e}")
    print("   📝 This is expected without proper TimesFM integration")

  # Step 9: Save ICF checkpoint
  print("\n9. Saving ICF checkpoint...")
  try:
    checkpoint_manager = ICFCheckpointManager("./examples/checkpoints")

    # Save ICF components (without base model)
    icf_checkpoint_path = checkpoint_manager.save_icf_checkpoint(
      model=model,
      checkpoint_name="basic_forecasting_example",
      description="Basic forecasting example with 3 covariates",
      metadata={
        "examples_count": len(examples),
        "max_ts_length": max(len(ex.time_series) for ex in examples),
        "experiment": "basic_usage_demo",
      },
    )
    print(f"   ✓ ICF checkpoint saved: {icf_checkpoint_path}")

    # List available checkpoints
    available = checkpoint_manager.list_checkpoints()
    print(f"   📋 Available checkpoints: {available}")

  except Exception as e:
    print(f"   ⚠️  Checkpoint saving failed: {e}")

  # Step 10: Load checkpoint example
  print("\n10. Loading checkpoint example...")
  try:
    # Create ICF checkpoint info
    icf_checkpoint = ICFCheckpoint(
      base_timesfm_checkpoint={"huggingface_repo_id": "google/timesfm-1.0-200m"},
      icf_checkpoint_path="./examples/checkpoints/icf/basic_forecasting_example.pt",
      enabled_tasks=["forecasting"],
      icf_config=config,
      description="Loaded from basic usage example",
    )

    # Load model from checkpoint
    loaded_model = checkpoint_manager.load_from_checkpoint(icf_checkpoint)
    print("   ✓ Model loaded from checkpoint")
    print(f"   📊 Loaded model info: {loaded_model.get_model_info()}")

  except Exception as e:
    print(f"   ⚠️  Checkpoint loading failed: {e}")

  # Step 11: Model information summary
  print("\n11. Model Summary")
  print("-" * 30)
  info = model.get_model_info()
  for key, value in info.items():
    print(f"   {key}: {value}")

  print("\n🎉 Basic usage example completed!")
  print("=" * 50)


if __name__ == "__main__":
  main()
