# TimesFM-ICF Package Setup and Testing Guide

This guide walks through setting up and testing the TimesFM-ICF package.

## Package Structure

```
timesfm-icf/
├── pyproject.toml          # Package configuration and dependencies
├── README.md               # Package documentation
├── LICENSE                 # MIT License
├── src/
│   └── timesfm_icf/        # Main package source
│       ├── __init__.py     # Package exports
│       ├── token_manager.py        # Token management system
│       ├── sequence_builder.py     # Sequence building with templates
│       ├── components.py          # ICF-specific components  
│       ├── icf_model.py          # Main TimesFMICF wrapper
│       ├── model_config.py       # Model configuration
│       ├── training_config.py    # Training configuration
│       └── checkpoint_manager.py # Checkpoint management
├── examples/               # Usage examples
│   ├── basic_usage.py         # Basic API demonstration
│   ├── classification_finetune.py  # Classification fine-tuning
│   └── forecasting_finetune.py    # Forecasting fine-tuning
├── tests/                  # Unit tests (future)
└── test_package.py        # Basic import/initialization tests
```

## Installation Steps

### 1. Navigate to Package Directory
```bash
cd /workspace/timesfm/timesfm-icf
```

### 2. Install Package in Development Mode
```bash
# Install package with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 3. Install TimesFM Dependency
```bash
# Install Google's TimesFM package
pip install timesfm

# Or install specific version
pip install timesfm>=1.0.0
```

## Testing the Package

### 1. Basic Import Test
```bash
python test_package.py
```

### 2. Run Examples
```bash
# Basic usage (requires TimesFM installed)
python examples/basic_usage.py

# Classification fine-tuning demo
python examples/classification_finetune.py

# Forecasting fine-tuning demo  
python examples/forecasting_finetune.py
```

## Expected Output from test_package.py

```
🧪 TimesFM-ICF Package Tests
========================================
🔍 Testing imports...
   ✓ Token management components
   ✓ Sequence builder components
   ✓ ICF components
   ✓ Main ICF model
   ✓ Configuration classes
   ✓ Checkpoint manager

📦 Testing package import...
   ✓ Package imported successfully

🏗️ Testing core initialization...
   ✓ Core tokens: 2 tokens
   ✓ Required tokens present
   ✓ Default config: ['forecasting']
   ✓ Custom config: 5 max examples

📊 Testing example creation...
   ✓ Example created - TS length: 100
   ✓ Covariates shape: (3,)
   ✓ Classification example - Label: 1

✨ Test Summary
--------------------
Package structure: ✓
Core components: ✓
Configuration: ✓
Examples: ✓

🎉 All basic tests passed!
```

## Core Components Overview

### Token Management
- **CoreTokens**: Defines COVARIATE_SEP, EXAMPLE_SEP tokens
- **TaskTokens**: Manages task-specific separator tokens  
- **ICFTokenManager**: Handles vocabulary extension and token mapping

### Sequence Building
- **ICFExample**: Data structure for time series with optional covariates/labels
- **CompiledSequenceBuilder**: Creates fixed-size sequences with attention masking

### ICF Components
- **ICFComponents**: Learnable separator embeddings and task heads
- **ClassificationHead**: Task-specific classification head with pooling

### Main Model
- **TimesFMICF**: Primary wrapper integrating TimesFM with ICF capabilities
- Supports model loading, compilation, forecasting, and classification

### Configuration
- **ICFConfig**: Model configuration (tasks, sequence lengths, templates)
- **ICFTrainingConfig**: Training parameters and optimization settings

### Checkpoint Management
- **ICFCheckpointManager**: Save/load ICF models with metadata tracking
- Supports both full model and ICF-only checkpoints

## Key Features

### 1. Independent Package Design
- Uses TimesFM as dependency without modifying it
- Composition-based architecture for clean integration
- Maintains compatibility with TimesFM updates

### 2. Multi-task Architecture  
- Unified framework for forecasting and classification
- Extensible design for adding new tasks
- Task-specific heads and loss functions

### 3. torch.compile Optimization
- Pre-allocated token slots for static sequence shapes
- Attention masking for variable content lengths
- Compilation-friendly design patterns

### 4. Advanced Token System
- Core shared separator tokens (COVARIATE_SEP, EXAMPLE_SEP)
- Task-specific extensions (CLASSIFICATION_SEP, etc.)
- Dynamic vocabulary management

### 5. Comprehensive Checkpointing
- Full model checkpoints with TimesFM base model
- ICF-only checkpoints for lighter storage
- Rich metadata tracking for experiment management

## Troubleshooting

### Import Errors
If you see import errors, ensure:
1. Package is installed: `pip install -e .`
2. TimesFM is available: `pip install timesfm`
3. Dependencies are met: `pip install torch numpy`

### TimesFM Loading Issues
If TimesFM checkpoint loading fails:
1. Check internet connection for Hugging Face download
2. Verify timesfm package version: `pip show timesfm`  
3. Use mock mode in examples for testing without TimesFM

### torch.compile Issues
If compilation fails:
1. Update PyTorch: `pip install torch>=2.1.0`
2. Disable compilation in config: `enable_torch_compile=False`
3. Check CUDA compatibility for GPU usage

## Next Steps

1. **Install Dependencies**: Run the installation steps above
2. **Test Basic Functionality**: Execute `python test_package.py`
3. **Try Examples**: Run the example scripts to see the API in action
4. **Implement Your Use Case**: Use the examples as templates for your specific needs
5. **Contribute**: Add new tasks, improve documentation, or submit bug fixes

## Package API Summary

```python
from timesfm_icf import TimesFMICF, ICFConfig, ICFExample

# Configure model
config = ICFConfig(
    enabled_tasks=['forecasting', 'classification'],
    max_examples=4,
    total_sequence_length=64
)

# Initialize model
model = TimesFMICF(config=config)

# Load TimesFM and initialize ICF
model.load_timesfm_checkpoint("google/timesfm-1.0-200m")
model.initialize_icf_components(['forecasting'])

# Create examples
examples = [
    ICFExample(time_series=ts_data, covariates=cov_data)
    for ts_data, cov_data in zip(time_series_list, covariate_list)
]

# Forecast
predictions = model.forecast(examples, horizon=24)

# Save model
from timesfm_icf import ICFCheckpointManager
checkpoint_manager = ICFCheckpointManager("./checkpoints")
checkpoint_manager.save_full_checkpoint(model, "my_model")
```