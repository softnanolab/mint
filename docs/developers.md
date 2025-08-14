# Developer Standards

## Argument Management with Fire

All Python scripts must use [Python Fire](https://github.com/google/python-fire) for argument management.

### Example

```python
import fire

def main(
    input_file: str,
    output_dir: str,
    num_workers: int = 4,
    verbose: bool = False
) -> None:
    """Process data with specified parameters."""
    # Implementation here
    pass

if __name__ == "__main__":
    fire.Fire(main)
```

**Usage:**
```bash
python script.py input.txt output/ --num_workers=8 --verbose
python script.py --help
```

### Migration from argparse

**Before:**
```python
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
```

**After:**
```python
import fire
def main(input: str) -> None:
    """Process data."""
    pass
if __name__ == "__main__":
    fire.Fire(main)
```

## Configuration with Hydra

Use [Hydra](https://hydra.cc/) for complex configuration management.

### Current Setup

```yaml
# src/mint/configs/main.yaml
meta:
  data_dir: ${oc.env:DATA_DIR}
  experiment_name: mint_test_run

wandb:
  project: "${meta.experiment_name}"
  entity: ${oc.env:WANDB_ENTITY}
```

### Usage

```bash
# Override config values
python src/mint/train.py meta.experiment_name=new_experiment

# Hyperparameter search
python src/mint/train.py -m training_args.lr=1e-4,1e-3,1e-2
```

## Experiment Logging with W&B

Integrate [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Current Integration

```python
# src/mint/train.py
from lightning.pytorch.loggers import WandbLogger

wdb_logger = WandbLogger(
    name=cfg.wandb.name,
    project=cfg.wandb.project,
    entity=cfg.wandb.entity,
)
```

### Custom Logging

```python
import wandb

# Log metrics
wandb.log({"accuracy": 0.95, "epoch": current_epoch})

# Log artifacts
artifact = wandb.Artifact("model", type="model")
artifact.add_file("checkpoint.ckpt")
wandb.log_artifact(artifact)
```

## Code Standards

### Script Template

```python
#!/usr/bin/env python3
"""Script description."""

import fire
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def main(input_file: str, output_dir: str) -> None:
    """Main function description."""
    pass

if __name__ == "__main__":
    fire.Fire(main)
```

### Requirements

- **Type Hints**: Required for all function parameters
- **Docstrings**: Google-style for public functions
- **Line Length**: 99 characters (Black default)
- **Imports**: Grouped and sorted with isort

## Quick Start

```bash
# Setup
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .[dev]

# Environment
cp env.example .env
# Edit .env with your DATA_DIR

# Quality checks
pre-commit install
pytest
```

**Important**: Always run `uv pip install -e .[dev]` after pulling changes from GitHub to ensure your environment is up to date with the latest dependencies.

## Current Status

✅ **Fire**: `generate_pdb_ids.py`, `process_data.py`, `cath_processor.py`  
✅ **Hydra**: `src/mint/train.py` with `main.yaml`  
✅ **W&B**: Integrated in training pipeline
