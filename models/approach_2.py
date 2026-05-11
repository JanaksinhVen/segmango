"""
Main training script for SegFormer-based yield prediction.

Usage:
    python approach_2.py --weather --scale --fold 0 --variant b1 --unfreez_epoch 30 --batch_size 8
"""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from env_config import DATA_DIR
from env_config import ROOT_DIR
from dataset import FruitDataset, get_data_transforms, get_dataframes_from_splits
from models import SegFormerRegressor
from train import Trainer
from utils import (
    plot_training_curves, remove_module_prefix, load_checkpoint,
    save_model_and_scaler, prepare_filtered_data, find_latest_checkpoint,
    MetricsTracker
)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SegFormer regression model for yield prediction"
    )
    parser.add_argument('--weather', action='store_true', 
                       help='Include weather features')
    parser.add_argument('--scale', action='store_true', 
                       help='Include scale features')
    parser.add_argument('--fold', type=int, required=True, 
                       help='Fold number for cross-validation')
    parser.add_argument('--variant', type=str, required=True, 
                       help='Model variant (b0 or b1)')
    parser.add_argument('--unfreez_epoch', type=int, default=30, 
                       help='Epoch to unfreeze encoder')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override root directory for CSV and split data')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Override root directory for image folders')
    parser.add_argument('--isotree_split_dir', type=str, default=None,
                       help='Override directory for isotree split TXT files')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Override checkpoint save directory')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Override results directory for plots')
    parser.add_argument('--segformer_ckpt_dir', type=str, default=None,
                       help='Override folder or file for SegFormer checkpoints')
    parser.add_argument('--segformer_ckpt_path', type=str, default=None,
                       help='Explicit SegFormer checkpoint file path')
    
    return parser.parse_args()


def build_feature_list(args):
    """Build feature list based on arguments."""
    features = list(config.BASE_FEATURES)
    
    if args.weather:
        features.extend(config.WEATHER_FEATURES)
    if args.scale:
        features.extend(config.SCALE_FEATURES)
    
    return features


def resolve_data_paths(args):
    """Resolve CSV, image, and split directories from args or environment."""
    script_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = args.data_dir or DATA_DIR or os.path.join(script_root, 'data')
    data_root = os.path.abspath(data_root)

    data_2024_path = os.path.join(
        data_root, 'mlp_all_data_with_time_weather_scale_treewise_2024.csv'
    )
    data_2025_path = os.path.join(
        data_root, 'mlp_all_data_with_time_weather_scale_treewise_2025.csv'
    )
    if not os.path.exists(data_2024_path) or not os.path.exists(data_2025_path):
        fallback_data_root = os.path.join(script_root, 'data')
        if fallback_data_root != data_root:
            data_2024_path = os.path.join(
                fallback_data_root, 'mlp_all_data_with_time_weather_scale_treewise_2024.csv'
            )
            data_2025_path = os.path.join(
                fallback_data_root, 'mlp_all_data_with_time_weather_scale_treewise_2025.csv'
            )
            data_root = fallback_data_root

    image_root = args.image_dir or DATA_DIR or data_root
    image_folders = [
        os.path.join(image_root, 'Dataset_images_2024'),
        os.path.join(image_root, 'Dataset_images_2025')
    ]

    isotree_split_dir = args.isotree_split_dir or os.path.join(data_root, 'train_test_splits')

    return data_root, data_2024_path, data_2025_path, image_folders, isotree_split_dir


def get_checkpoint_paths(args, feature_length):
    """Get paths for saving models and plots."""
    variant_suffix = f"_{args.variant}" if args.variant == 'b0' else ""
    
    checkpoint_dir = args.checkpoint_dir or config.get_checkpoint_dir()
    results_dir = args.results_dir or config.get_results_dir()
    
    model_path = os.path.join(
        checkpoint_dir,
        f"fold_seg_reg_{args.fold}_{feature_length}{variant_suffix}_attention.pth"
    )
    plot_path = os.path.join(
        results_dir,
        f"fold_seg_reg_{args.fold}_{feature_length}{variant_suffix}_attention.png"
    )
    scaler_path = os.path.join(
        checkpoint_dir,
        f"S_fold_seg_reg_{args.fold}_{feature_length}{variant_suffix}_attention.pkl"
    )
    
    return model_path, plot_path, scaler_path


def get_segformer_checkpoint(args):
    """Find latest SegFormer checkpoint."""
    if args.segformer_ckpt_path:
        if not os.path.exists(args.segformer_ckpt_path):
            raise FileNotFoundError(f"SegFormer checkpoint not found: {args.segformer_ckpt_path}")
        return args.segformer_ckpt_path

    if args.segformer_ckpt_dir:
        if os.path.isfile(args.segformer_ckpt_dir):
            return args.segformer_ckpt_dir
        if os.path.isdir(args.segformer_ckpt_dir):
            return find_latest_checkpoint(args.segformer_ckpt_dir)
        raise FileNotFoundError(f"SegFormer checkpoint directory not found: {args.segformer_ckpt_dir}")

    if not DATA_DIR:
        raise ValueError("DATA_DIR environment variable not set")
    
    fold_suffix = 2 if args.variant == 'b0' else 2
    base_dir = DATA_DIR.replace('/Dataset_images_2024', '').replace('/Dataset_images_2025', '')
    folder = os.path.join(
        base_dir,
        f"work_dirs/segformer_768_sbatch_1_fold_{fold_suffix}_{args.variant}"
    )
    
    if not os.path.exists(folder):
        # Try without variant suffix
        folder = os.path.join(
            ROOT_DIR,
            # f"work_dirs/segformer_768_sbatch_1_fold_{fold_suffix}"
            "data/Model_weights/approach-1/segformer/"
        )
    
    return find_latest_checkpoint(folder)


def setup_data(args, feature_columns):
    """Prepare train/val/test data."""
    data_root, data_2024_path, data_2025_path, image_folders, isotree_split_dir = resolve_data_paths(args)

    # Load and filter base data
    full_df = prepare_filtered_data(
        data_2024_path,
        data_2025_path
    )
    full_df.to_csv('data_seg_reg.csv', index=False)
    
    # Load splits
    train_file = os.path.join(
        isotree_split_dir,
        f"train_{args.fold}.txt"
    )
    val_file = os.path.join(
        isotree_split_dir,
        f"val_{args.fold}.txt"
    )
    test_file = os.path.join(
        isotree_split_dir,
        f"test_{args.fold}.txt"
    )
    
    # train_df, val_df, test_df = get_dataframes_from_splits(
    #     full_df, train_file, val_file, test_file
    # )
    train_df = pd.read_csv(os.path.join(
        isotree_split_dir,
        f"train_split_{args.fold}.csv".format(args.fold))
    )
    val_df = pd.read_csv(os.path.join(
        isotree_split_dir,
        f"val_split_{args.fold}.csv".format(args.fold))
    )
    test_df = pd.read_csv(os.path.join(
        isotree_split_dir,
        f"test_split.csv".format(args.fold))
    )   
    
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save splits
    train_df.to_csv(f'train_split_{args.fold}.csv', index=False)
    val_df.to_csv(f'val_split_{args.fold}.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    
    return train_df, val_df, test_df, scaler


def setup_dataloaders(args, train_df, val_df, test_df, feature_columns, scaler, image_folders):
    """Create data loaders."""
    train_transform, val_test_transform = get_data_transforms(config.IMAGE_SIZE)
    
    train_dataset = FruitDataset(
        'train_split_{}.csv'.format(args.fold),
        image_folders,
        feature_columns,
        transform=train_transform,
        scaler=scaler
    )
    
    val_dataset = FruitDataset(
        'val_split_{}.csv'.format(args.fold),
        image_folders,
        feature_columns,
        transform=val_test_transform,
        scaler=scaler
    )
    
    test_dataset = FruitDataset(
        'test_split.csv',
        image_folders,
        feature_columns,
        transform=val_test_transform,
        scaler=scaler
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.DATA_NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.DATA_NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.DATA_NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def setup_model_and_training(model, device):
    """Setup model, optimizer, and scheduler."""
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"[GPU] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE, verbose=True
    )
    
    return model, criterion, optimizer, scheduler


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup
    print("[INFO] Building feature list...")
    feature_columns = build_feature_list(args)
    feature_length = len(feature_columns)
    
    print(f"[INFO] Features ({feature_length}): {feature_columns}")
    print(f"[INFO] Fold: {args.fold}, Variant: {args.variant}")
    
    # Get checkpoint paths
    model_path, plot_path, scaler_path = get_checkpoint_paths(args, feature_length)
    
    # Create output directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print(f"[DEVICE] {torch.cuda.get_device_name(0)}")
    
    # Prepare data
    print("[DATA] Preparing data...")
    train_df, val_df, test_df, scaler = setup_data(args, feature_columns)
    
    _, _, _, image_folders, _ = resolve_data_paths(args)
    print("[DATA] Creating data loaders...")
    train_loader, val_loader, test_loader = setup_dataloaders(
        args, train_df, val_df, test_df, feature_columns, scaler, image_folders
    )
    
    # Load encoder checkpoint
    print("[MODEL] Loading SegFormer encoder...")
    encoder_ckpt = get_segformer_checkpoint(args)
    
    # Build model
    print("[MODEL] Building SegFormer regressor...")
    model = SegFormerRegressor(
        encoder_ckpt=encoder_ckpt,
        variant=args.variant,
        num_extra_feats=len(feature_columns),
        hidden_dim=config.MODEL_HIDDEN_DIM,
        freeze_encoder=True
    )
    
    # Setup training
    model, criterion, optimizer, scheduler = setup_model_and_training(model, device)
    
    # Create trainer
    print("[TRAIN] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        early_stop_patience=config.EARLY_STOP_PATIENCE
    )
    
    # Train
    print("[TRAIN] Starting training...")
    metrics = trainer.train(
        num_epochs=config.NUM_EPOCHS,
        unfreeze_epoch=args.unfreez_epoch,
        model_save_path=model_path
    )
    
    # Save model and scaler
    print("[SAVE] Saving model and scaler...")
    save_model_and_scaler(model, scaler, model_path, scaler_path)
    
    # Plot training curves
    print("[PLOT] Plotting training curves...")
    metrics.plot_and_save(plot_path)
    
    # Test
    print("[TEST] Running test evaluation...")
    test_loss, test_preds, test_targets = trainer.test_epoch()
    
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\n[TEST] MSE: {test_loss:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    print("[DONE] Training complete!")


if __name__ == "__main__":
    main()
