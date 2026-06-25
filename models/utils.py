"""
Utility functions for training, evaluation, and data processing.
"""

import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def plot_training_curves(train_losses, val_losses, train_maes, val_maes,
                         train_r2s, val_r2s, save_path="training_curves.png"):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_losses, val_losses: Lists of training/validation losses
        train_maes, val_maes: Lists of training/validation MAE
        train_r2s, val_r2s: Lists of training/validation R² scores
        save_path: Path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_maes, label='Train MAE', color='green')
    plt.plot(epochs, val_maes, label='Val MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training & Validation MAE')
    plt.legend()
    plt.grid(True)

    # R² Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_r2s, label='Train R²', color='purple')
    plt.plot(epochs, val_r2s, label='Val R²', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training & Validation R² Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Training curves saved to: {save_path}")


def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state dict keys (from DataParallel wrapper).
    
    Args:
        state_dict: PyTorch state dictionary
        
    Returns:
        Cleaned state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def load_checkpoint(model, checkpoint_path, device):
    """
    Load checkpoint with 'module.' prefix handling.
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Loaded model
    """
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    cleaned_state_dict = remove_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    print("✅ Loaded checkpoint with cleaned keys.")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    return model.to(device)


def save_model_and_scaler(model, scaler, model_path, scaler_path):
    """
    Save model checkpoint and feature scaler.
    
    Args:
        model: PyTorch model
        scaler: Fitted StandardScaler
        model_path: Path to save model
        scaler_path: Path to save scaler
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")


def prepare_filtered_data(data_2024_path, data_2025_path):
    """
    Prepare filtered dataframes for 2024 and 2025 data.
    
    Args:
        data_2024_path: Path to 2024 CSV
        data_2025_path: Path to 2025 CSV
        
    Returns:
        Concatenated filtered dataframe
    """
    # Load 2024 data
    df_2024 = pd.read_csv(data_2024_path)
    target_filenames_2024 = [
        f"08_{i:02d}_{j:02d}" for i in range(1, 13) for j in range(1, 9)
    ]
    filtered_df_2024 = df_2024[
        df_2024['image_name_o'].isin(target_filenames_2024)
    ].reset_index(drop=True)
    
    # Load 2025 data
    df_2025 = pd.read_csv(data_2025_path)
    target_filenames_2025 = [
        f"N_03_{i:02d}_{j:02d}" for i in range(1, 21) for j in range(1, 9)
    ]
    filtered_df_2025 = df_2025[
        df_2025['image_name_o'].isin(target_filenames_2025)
    ].reset_index(drop=True)
    
    return pd.concat([filtered_df_2024, filtered_df_2025]).reset_index(drop=True)


def find_latest_checkpoint(folder, pattern="best_mIoU_iter_"):
    """
    Find latest checkpoint in folder.
    
    Args:
        folder: Folder containing checkpoints
        pattern: Filename pattern to search
        
    Returns:
        Full path to latest checkpoint
    """
    all_files = os.listdir(folder)
    pth_files = [
        f for f in all_files 
        if f.startswith(pattern) and f.endswith(".pth")
    ]
    
    if not pth_files:
        raise FileNotFoundError(f"No checkpoints matching {pattern} found in {folder}")
    
    # Sort by iteration number
    pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = os.path.join(folder, pth_files[-1])
    
    print(f"[INFO] Using checkpoint: {latest}")
    return latest


class MetricsTracker:
    """Tracks and stores training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_r2s = []
        self.val_r2s = []
    
    def update_train(self, loss, predictions, targets):
        """Update training metrics."""
        self.train_losses.append(loss)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        self.train_maes.append(mae)
        self.train_r2s.append(r2)
    
    def update_val(self, loss, predictions, targets):
        """Update validation metrics."""
        self.val_losses.append(loss)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        self.val_maes.append(mae)
        self.val_r2s.append(r2)
    
    def get_epoch_summary(self, epoch):
        """Get formatted summary of metrics for an epoch."""
        idx = epoch - 1
        return (
            f"Epoch [{epoch}] "
            f"Train MSE: {self.train_losses[idx]:.4f}, "
            f"MAE: {self.train_maes[idx]:.4f}, "
            f"R²: {self.train_r2s[idx]:.4f} | "
            f"Val MSE: {self.val_losses[idx]:.4f}, "
            f"MAE: {self.val_maes[idx]:.4f}, "
            f"R²: {self.val_r2s[idx]:.4f}"
        )
    
    def plot_and_save(self, save_path):
        """Plot and save training curves."""
        plot_training_curves(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_maes=self.train_maes,
            val_maes=self.val_maes,
            train_r2s=self.train_r2s,
            val_r2s=self.val_r2s,
            save_path=save_path
        )
