# SegFormer Yield Prediction - Modularized Code

This directory contains the modularized training pipeline for the SegFormer-based mango yield prediction model.

## Configuration System

The project uses a **two-tier configuration system** following best practices:

### 1. Environment Configuration (.env)
System-specific and sensitive settings are loaded from the `.env` file:
- `ROOT_DIR`: Project root directory (required)
- `DATA_DIR`: Data directory with images and preprocessed data (required)
- `CHECKPOINT_DIR`: (Optional) Custom checkpoint save directory
- `RESULTS_DIR`: (Optional) Custom results directory
- `NUM_WORKERS`: (Optional) Number of workers for data loading (default: 4)
- `RANDOM_SEED`: (Optional) Random seed for reproducibility (default: 42)

**Setup:**
```bash
cp .env.example .env
# Edit .env with your system-specific paths
```

### 2. Model Configuration (config.py)
Model hyperparameters, feature definitions, and training constants are defined in `config.py`:
- Training hyperparameters (learning rate, batch size, epochs)
- Model architecture parameters (hidden dims, image size)
- Feature definitions (weather, scale features)
- Dynamic path getters with intelligent fallbacks

## File Structure

- **`env_config.py`**: Environment variable loader with validation
- **`config.py`**: Model configuration and hyperparameters
- **`dataset.py`**: Dataset classes and data loading utilities
- **`models.py`**: Model architectures (SegFormerRegressor, SelfAttentionFusion)
- **`train.py`**: Training loop and Trainer class
- **`utils.py`**: Utility functions (plotting, checkpointing, metrics tracking)
- **`approach_2.py`**: Main training script (entry point)
- **`.env.example`**: Template for environment configuration

## Quick Start

### Prerequisites
- PyTorch with CUDA support
- MMSegmentation (`mmseg`)
- Albumentations
- Scikit-learn
- python-dotenv
- All other dependencies in requirements.txt

### Setup Environment

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your system paths:
```bash
ROOT_DIR=/path/to/segmango_ssh
DATA_DIR=/path/to/dataset_images
NUM_WORKERS=4
RANDOM_SEED=42
```

### Running Training

```bash
python approach_2.py \
  --tree_n 1 \
  --new_tree \
  --weather \
  --scale \
  --fold 0 \
  --variant b1 \
  --unfreez_epoch 30 \
  --batch_size 8
```

### Arguments

- `--tree_n`: Tree number for iso-tree validation (required)
- `--new_tree`: Include new tree data (optional flag)
- `--weather`: Include weather features (optional flag)
- `--scale`: Include scale features (optional flag)
- `--fold`: Fold number for cross-validation (required)
- `--variant`: Model variant - 'b0' or 'b1' (required)
- `--unfreez_epoch`: Epoch to unfreeze encoder (default: 30)
- `--batch_size`: Training batch size (default: 8)

## Module Descriptions

### env_config.py
Environment variable loader with validation:
- `EnvConfig` class with static methods for all variables
- Validates that required paths exist
- Provides fallback defaults for optional variables
- Automatically loads `.env` on import via `load_dotenv()`

Key features:
- Required variables: `ROOT_DIR`, `DATA_DIR`
- Optional variables: `CHECKPOINT_DIR`, `RESULTS_DIR`, `NUM_WORKERS`, `RANDOM_SEED`
- Raises `ValueError` if required variables missing or paths invalid

### config.py
Model configuration with three responsibilities:
1. **Load environment** via `env_config.py`
2. **Define hyperparameters** (learning rates, epochs, features)
3. **Provide dynamic path getters** with intelligent fallbacks

Dynamic path getters:
- If custom path set in `.env`, uses it
- Otherwise, constructs default from `ROOT_DIR` or `DATA_DIR`
- Falls back to relative paths if environment unavailable

Functions:
- `get_checkpoint_dir()`, `get_results_dir()`
- `get_isotree_split_dir()`, `get_data_2024_path()`, `get_data_2025_path()`
- `get_image_folders()`

### dataset.py
- `FruitDataset`: Custom PyTorch Dataset class
  - Handles image loading and tabular feature scaling
  - Supports multiple image folders and formats
  
- `get_data_transforms()`: Creates augmentation pipelines
  - Uses Albumentations for efficient augmentation
  - Training: resize, flip, color jitter
  - Validation/Test: resize only
  
- `get_dataframes_from_splits()`: Loads data splits from text files

### models.py
- `SelfAttentionFusion`: Fusion module for combining visual and tabular features
  - Learnable weighted combination (alpha, beta parameters)
  - Self-attention mechanism for feature interaction
  
- `SegFormerRegressor`: Main model
  - Supports variants b0 and b1 with different encoder dimensions
  - Encoder freezing for staged fine-tuning
  - Attention-based feature fusion

### train.py
- `Trainer`: High-level training manager
  - Handles train/val/test loops
  - Implements early stopping
  - Tracks metrics automatically
  - Methods: `train_epoch()`, `val_epoch()`, `test_epoch()`, `train()`

### utils.py
- `plot_training_curves()`: Visualize training progress
- `remove_module_prefix()`: Handle DataParallel state dicts
- `load_checkpoint()`: Load model checkpoints with proper error handling
- `save_model_and_scaler()`: Save model and feature scaler
- `MetricsTracker`: Automatic metrics tracking and summary generation
- `find_latest_checkpoint()`: Find latest checkpoint in a directory

### approach_2.py
Main entry point that orchestrates:
1. Argument parsing
2. Feature list construction
3. Data preparation and loading
4. Model initialization
5. Training pipeline
6. Test evaluation
7. Result saving

## Configuration Examples

### Example 1: Minimal Configuration
```env
ROOT_DIR=/home/user/segmango_ssh
DATA_DIR=/mnt/data/images
```
- Checkpoints: `{ROOT_DIR}/checkpoints/folds/`
- Results: `{ROOT_DIR}/results/`
- Workers: 4, Seed: 42

### Example 2: Custom Paths
```env
ROOT_DIR=/home/user/segmango_ssh
DATA_DIR=/mnt/data/images
CHECKPOINT_DIR=/mnt/fast_ssd/checkpoints
RESULTS_DIR=/mnt/results
NUM_WORKERS=8
RANDOM_SEED=123
```

### Example 3: Cloud Deployment
```env
ROOT_DIR=/workspace/segmango_ssh
DATA_DIR=/data/dataset_images
CHECKPOINT_DIR=/mnt/checkpoints
RESULTS_DIR=/mnt/results
NUM_WORKERS=16
CUDA_DEVICE=0
```

## Best Practices

1. **Never commit .env**: Already in `.gitignore`
2. **Use .env.example**: Template for documentation
3. **Validate paths**: Script checks paths exist on startup
4. **Environment override**: System env vars override .env file
5. **Configuration immutability**: Config loaded once at startup

## Error Handling

If environment variables missing:
```
[WARNING] ROOT_DIR not found in environment.
Please set it in /path/to/.env or export ROOT_DIR environment variable.
```

To fix:
```bash
# Create .env from template
cp .env.example .env

# Edit with your paths
vim .env

# Verify
python -c "from env_config import ROOT_DIR, DATA_DIR; print(f'ROOT_DIR: {ROOT_DIR}'); print(f'DATA_DIR: {DATA_DIR}')"
```

## Extending the Code

### Adding environment variables
1. Add getter method in `env_config.py`
2. Import and use in `config.py`
3. Document in `.env.example`

### Adding new model variant
1. Edit `config.py` to add variant config
2. Modify `SegFormerRegressor.__init__()` in `models.py`

### Custom data augmentation
Edit `get_data_transforms()` in `dataset.py`

### Different fusion strategies
Create new fusion module in `models.py` inheriting from `nn.Module`

### Custom loss functions
Pass different `criterion` to `Trainer` in `approach_2.py`

## Output

After training, saved files:
- **Model**: `{CHECKPOINT_DIR}/fold_seg_reg_[fold]_[features]_[variant]_attention.pth`
- **Scaler**: `{CHECKPOINT_DIR}/S_fold_seg_reg_[fold]_[features]_[variant]_attention.pkl`
- **Plots**: `{RESULTS_DIR}/fold_seg_reg_[fold]_[features]_[variant]_attention.png`

## Example Output

```
[INFO] Using SegFormer-B1 encoder with output dim 512
[DATA] Train: 450, Val: 120, Test: 80
[TRAIN] Starting training...
Epoch [1] Train MSE: 125.3456, MAE: 9.2345, R²: 0.2456 | Val MSE: 135.2345, MAE: 10.1234, R²: 0.1890
...
[TEST] MSE: 128.5432, MAE: 9.8765, R²: 0.2234
[DONE] Training complete!
```

## Troubleshooting

### Missing .env file
```bash
cp .env.example .env
vim .env  # Edit with your paths
```

### Verify environment variables
```python
from env_config import ROOT_DIR, DATA_DIR
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
```

### Check path validation
```bash
ls -la /path/to/ROOT_DIR
ls -la /path/to/DATA_DIR
```

