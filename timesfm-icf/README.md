# TimesFM-ICF: In-Context Fine-Tuning for TimesFM

TimesFM-ICF is an independent extension package that adds In-Context Fine-Tuning capabilities to Google's TimesFM foundation model using learnable separator tokens.

## Features

- 🚀 **Independent Package**: Uses TimesFM as a dependency without modifying the original codebase
- 🎯 **Multi-Task Learning**: Support for forecasting, classification, and extensible task types
- ⚡ **torch.compile Compatible**: Pre-allocated token slots ensure fixed input shapes for optimal compilation
- 🔧 **Flexible Fine-Tuning**: Task-specific and multi-task training capabilities
- 💾 **Robust Checkpointing**: Save and load both base TimesFM and fine-tuned ICF models
- 🎨 **Extensible Token System**: Core tokens + task-specific extensions

## Quick Start

```python
import torch
from timesfm_icf import TimesFMICF, ICFExample
from timesfm import TimesFmCheckpoint

# Load pretrained TimesFM + initialize ICF
model = TimesFMICF()
model.load_timesfm_checkpoint(TimesFmCheckpoint(
    huggingface_repo_id="google/timesfm-1.0-200m"
))
model.initialize_icf_components(enabled_tasks=['forecasting'])
model.compile_model()

# Prepare data with covariates
examples = [
    ICFExample(time_series=[1, 2, 3, 4, 5], covariates=[0.1, 0.2]),
    ICFExample(time_series=[2, 3, 4, 5, 6], covariates=[0.3, 0.4])
]

# Forecast
predictions = model.forecast(examples, horizon=24)
```

## Installation

```bash
# Install from source
git clone https://github.com/google-research/timesfm
cd timesfm/timesfm-icf
pip install -e .

# TimesFM will be installed automatically as a dependency
```

## Architecture

TimesFM-ICF extends TimesFM with:
- **Learnable Separator Tokens**: COVARIATE_SEP, EXAMPLE_SEP, CLASS_SEP
- **Pre-allocated Templates**: Fixed-size sequences for torch.compile compatibility  
- **Task-Specific Heads**: Classification, regression, and extensible architectures
- **ICF Components**: Trainable parameters separate from base TimesFM model

## License

Apache License 2.0