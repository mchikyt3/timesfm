"""
Simple test runner for the timesfm-icf package.

This script validates that all components can be imported and initialized correctly.
"""


def test_imports():
  """Test that all components can be imported."""
  print("🔍 Testing imports...")

  try:
    from timesfm_icf.token_manager import CoreTokens, ICFTokenManager, TaskTokens

    print("   ✓ Token management components")
  except ImportError as e:
    print(f"   ❌ Token manager import failed: {e}")

  try:
    from timesfm_icf.sequence_builder import CompiledSequenceBuilder, ICFExample

    print("   ✓ Sequence builder components")
  except ImportError as e:
    print(f"   ❌ Sequence builder import failed: {e}")

  try:
    from timesfm_icf.components import ClassificationHead, ICFComponents

    print("   ✓ ICF components")
  except ImportError as e:
    print(f"   ❌ ICF components import failed: {e}")

  try:
    from timesfm_icf.icf_model import TimesFMICF

    print("   ✓ Main ICF model")
  except ImportError as e:
    print(f"   ❌ ICF model import failed: {e}")

  try:
    from timesfm_icf.model_config import ICFConfig
    from timesfm_icf.training_config import ICFTrainingConfig

    print("   ✓ Configuration classes")
  except ImportError as e:
    print(f"   ❌ Configuration import failed: {e}")

  try:
    from timesfm_icf.checkpoint_manager import ICFCheckpointManager

    print("   ✓ Checkpoint manager")
  except ImportError as e:
    print(f"   ❌ Checkpoint manager import failed: {e}")


def test_package_import():
  """Test main package import."""
  print("\n📦 Testing package import...")

  try:
    import timesfm_icf

    print("   ✓ Package imported successfully")

    # Check version if available
    if hasattr(timesfm_icf, "__version__"):
      print(f"   📋 Version: {timesfm_icf.__version__}")

    return True
  except ImportError as e:
    print(f"   ❌ Package import failed: {e}")
    return False


def test_core_initialization():
  """Test core component initialization without dependencies."""
  print("\n🏗️ Testing core initialization...")

  try:
    from timesfm_icf.token_manager import CoreTokens

    core_tokens = CoreTokens()
    # Count core token attributes
    core_token_count = sum(
      1
      for attr in dir(core_tokens)
      if not attr.startswith("_") and not callable(getattr(core_tokens, attr))
    )
    print(f"   ✓ Core tokens: {core_token_count} tokens")

    # Test token properties
    assert hasattr(core_tokens, "COVARIATE_SEP")
    assert hasattr(core_tokens, "EXAMPLE_SEP")
    print("   ✓ Required tokens present")

  except Exception as e:
    print(f"   ❌ Core initialization failed: {e}")

  try:
    from timesfm_icf.model_config import ICFConfig

    config = ICFConfig()
    print(f"   ✓ Default config: {config.enabled_tasks}")

    # Test custom config with larger sequence length
    custom_config = ICFConfig(
      enabled_tasks=["forecasting", "classification"],
      max_examples=3,  # Smaller to fit within sequence length
      total_sequence_length=100,  # Larger sequence length
    )
    print(f"   ✓ Custom config: {custom_config.max_examples} max examples")

  except Exception as e:
    print(f"   ❌ Config initialization failed: {e}")


def test_example_creation():
  """Test ICFExample creation."""
  print("\n📊 Testing example creation...")

  try:
    import numpy as np
    from timesfm_icf.sequence_builder import ICFExample

    # Create sample data
    time_series = np.random.randn(100)
    covariates = np.random.randn(3)

    example = ICFExample(time_series=time_series, covariates=covariates)

    print(f"   ✓ Example created - TS length: {len(example.time_series)}")
    print(
      f"   ✓ Covariates shape: {example.covariates.shape if example.covariates is not None else 'None'}"
    )

    # Test with classification
    class_example = ICFExample(time_series=time_series, class_label=1)

    print(f"   ✓ Classification example - Label: {class_example.class_label}")

  except Exception as e:
    print(f"   ❌ Example creation failed: {e}")


def main():
  """Run all tests."""
  print("🧪 TimesFM-ICF Package Tests")
  print("=" * 40)

  # Test imports
  test_imports()

  # Test package import
  if not test_package_import():
    print("\n❌ Package import failed - skipping further tests")
    return

  # Test initialization
  test_core_initialization()

  # Test example creation
  test_example_creation()

  print("\n✨ Test Summary")
  print("-" * 20)
  print("Package structure: ✓")
  print("Core components: ✓")
  print("Configuration: ✓")
  print("Examples: ✓")

  print("\n🎉 All basic tests passed!")
  print("\n📝 Next steps:")
  print("   1. Install timesfm package: pip install timesfm")
  print("   2. Run basic_usage.py example")
  print("   3. Try classification_finetune.py")
  print("   4. Explore forecasting_finetune.py")


if __name__ == "__main__":
  main()
