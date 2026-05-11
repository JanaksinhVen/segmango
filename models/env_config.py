"""
Environment variable loader with validation and defaults.
Loads configuration from .env file and system environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_FILE, override=False)


class EnvConfig:
    """
    Environment configuration loader with validation.
    Handles paths and system-specific settings from .env file.
    """
    
    # ============= Required Environment Variables =============
    @staticmethod
    def get_root_dir():
        """Get project root directory."""
        root = os.getenv("ROOT_DIR")
        if not root:
            raise ValueError(
                "ROOT_DIR not found in environment. "
                f"Please set it in {ENV_FILE} or export ROOT_DIR environment variable."
            )
        if not os.path.exists(root):
            raise ValueError(f"ROOT_DIR path does not exist: {root}")
        return root
    
    @staticmethod
    def get_data_dir():
        """Get data directory for dataset images."""
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            raise ValueError(
                "DATA_DIR not found in environment. "
                f"Please set it in {ENV_FILE} or export DATA_DIR environment variable."
            )
        if not os.path.exists(data_dir):
            raise ValueError(f"DATA_DIR path does not exist: {data_dir}")
        return data_dir
    
    # ============= Optional Environment Variables =============
    @staticmethod
    def get_checkpoint_dir():
        """Get checkpoint save directory (optional, uses default if not set)."""
        return os.getenv("CHECKPOINT_DIR", None)
    
    @staticmethod
    def get_results_dir():
        """Get results directory for plots (optional, uses default if not set)."""
        return os.getenv("RESULTS_DIR", None)
    
    @staticmethod
    def get_cuda_device():
        """Get preferred CUDA device (optional)."""
        return os.getenv("CUDA_DEVICE", None)
    
    @staticmethod
    def get_num_workers():
        """Get number of workers for data loading (optional)."""
        return int(os.getenv("NUM_WORKERS", 4))
    
    @staticmethod
    def get_seed():
        """Get random seed (optional)."""
        return int(os.getenv("RANDOM_SEED", 42))


# Load environment variables on module import
try:
    ROOT_DIR = EnvConfig.get_root_dir()
    DATA_DIR = EnvConfig.get_data_dir()
except ValueError as e:
    print(f"[WARNING] {e}")
    ROOT_DIR = None
    DATA_DIR = None

# Optional configurations
CHECKPOINT_DIR = EnvConfig.get_checkpoint_dir()
RESULTS_DIR = EnvConfig.get_results_dir()
CUDA_DEVICE = EnvConfig.get_cuda_device()
NUM_WORKERS = EnvConfig.get_num_workers()
RANDOM_SEED = EnvConfig.get_seed()
