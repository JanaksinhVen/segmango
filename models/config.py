"""
Configuration and constants for the SegFormer regression model training.

This module defines model hyperparameters, feature definitions, and default values.
System-specific paths are loaded from .env file via env_config.py.
"""

import os
from pathlib import Path
from env_config import ROOT_DIR, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, NUM_WORKERS, RANDOM_SEED

# ============= Paths (Loaded from Environment) =============
# These are loaded from .env file. If custom paths are not set in .env,
# defaults are constructed from ROOT_DIR and DATA_DIR

def get_checkpoint_dir():
    """Get checkpoint directory, with fallback to default."""
    if CHECKPOINT_DIR:
        return CHECKPOINT_DIR
    if ROOT_DIR:
        return os.path.join(ROOT_DIR, "checkpoints", "folds")
    return "./checkpoints/folds"


def get_results_dir():
    """Get results directory, with fallback to default."""
    if RESULTS_DIR:
        return RESULTS_DIR
    if ROOT_DIR:
        return os.path.join(ROOT_DIR, "results")
    return "./results"


def get_isotree_split_dir():
    """Get isotree split directory."""
    if DATA_DIR:
        return os.path.join(DATA_DIR, "isotree_data_split")
    return "./isotree_data_split"


def get_data_2024_path():
    """Get 2024 data CSV path."""
    if DATA_DIR:
        return os.path.join(DATA_DIR, "mlp_all_data_with_time_weather_scale_treewise_2024.csv")
    return "./mlp_all_data_with_time_weather_scale_treewise_2024.csv"


def get_data_2025_path():
    """Get 2025 data CSV path."""
    if DATA_DIR:
        return os.path.join(DATA_DIR, "mlp_all_data_with_time_weather_scale_treewise_2025.csv")
    return "./mlp_all_data_with_time_weather_scale_treewise_2025.csv"


def get_image_folders():
    """Get image folder paths."""
    if DATA_DIR:
        return [
            os.path.join(DATA_DIR, "Dataset_images_2024"),
            os.path.join(DATA_DIR, "Dataset_images_2025")
        ]
    return ["./Dataset_images_2024", "./Dataset_images_2025"]


# Lazy-load paths on first access
_paths_cache = {}

def _get_path(key, getter):
    """Cache and return paths."""
    if key not in _paths_cache:
        _paths_cache[key] = getter()
    return _paths_cache[key]


# ============= Hyperparameters (Model Training) =============
IMAGE_SIZE = 768
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 30
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# ============= Feature Sets =============
BASE_FEATURES = ['time']
WEATHER_FEATURES = [
    'temp', 'dew', 'precip', 'precipprob', 'visibility',
    'solarradiation', 'severerisk', 'preciptype', 'winddir',
    'windgust', 'windspeed'
]
SCALE_FEATURES = ['scale_sum_r_o', 'scale_max_r_o', 'scale_std_r_o']
TARGET_COLUMN = 'n_fruit_o'

# ============= Model Configuration =============
MODEL_HIDDEN_DIM = 256
UNFREEZE_EPOCH_DEFAULT = 30

# ============= System Configuration =============
# Loaded from .env or defaults
DATA_NUM_WORKERS = NUM_WORKERS
RANDOM_SEED_VALUE = RANDOM_SEED
