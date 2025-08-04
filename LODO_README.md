# LODO (Leave-One-Domain-Out) Training Feature

## Overview

The LODO (Leave-One-Domain-Out) feature allows you to evaluate model generalization by training on all datasets except one, which is held out as a test set. This is useful for assessing how well the model generalizes to unseen domains.

## Usage

### Basic Command

```bash
uv run battery-foundation train --config configs/default.yaml --lodo HUST
```

This command will:
- Load all datasets from the config except `HUST`
- Use `HUST` as the test dataset
- Train the model on the remaining datasets
- Evaluate the model on `HUST` during training

### Available Datasets

Based on the default configuration, the following datasets are available for LODO:
- `HUST`
- `MATR`
- `CALCE`
- `NASA`
- `OX`
- `RWTH`
- `UCL`

### Examples

```bash
# Train with HUST as test set
uv run battery-foundation train --config configs/default.yaml --lodo HUST

# Train with NASA as test set
uv run battery-foundation train --config configs/default.yaml --lodo NASA

# Train with CALCE as test set
uv run battery-foundation train --config configs/default.yaml --lodo CALCE
```

## How It Works

1. **Dataset Splitting**: When `--lodo` is specified, the system automatically:
   - Excludes the specified dataset from training/validation
   - Uses the specified dataset as the test set

2. **Training Process**: During training, the model:
   - Trains on all datasets except the LODO dataset
   - Validates on the same training datasets (if validation split is implemented)
   - Evaluates on the LODO dataset after each epoch

3. **Monitoring**: The training logs will show:
   - Training loss on the training datasets
   - Validation loss (if available)
   - Test loss on the LODO dataset

## Output

The training will produce:
- Regular checkpoints saved during training
- Best model checkpoint based on validation loss
- Training history including test loss for the LODO dataset
- Logs showing the dataset splits and evaluation metrics

## Configuration

The LODO feature works with any existing configuration file. The `--lodo` argument overrides the dataset selection without modifying the original config file.

## Notes

- The LODO dataset is completely excluded from training to ensure proper domain generalization evaluation
- Test evaluation happens after each epoch alongside validation
- The feature is backward compatible - existing training commands will work without changes 