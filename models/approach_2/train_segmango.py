import argparse
import os   
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm   
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib

# Load variables from .env file
load_dotenv()

# Get the ROOT_DIR variable
data_base_path = os.getenv("DATA_DIR")
project_root_dir = os.getenv("ROOT_DIR")

from utils.dataset import FruitDataset
from utils.segmango_model import SegFormerRegressor

def plot_training_curves(train_losses, val_losses,
                         train_maes, val_maes,
                         train_r2s, val_r2s,
                         save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # --- Plot 1: Loss ---
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: MAE ---
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_maes, label='Train MAE', color='green')
    plt.plot(epochs, val_maes, label='Val MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training & Validation MAE')
    plt.legend()
    plt.grid(True)

    # --- Plot 3: R² Score ---
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



def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument('--weather', action='store_true', help='Include weather features')
    parser.add_argument('--scale', action='store_true', help='Include scale features')
    parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
    parser.add_argument('--variant', type=str, required=True, help='Model varient')
    parser.add_argument('--unfreez_epoch', type=int, default=30, help='Epoch to start unfreezing the encoder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    return parser.parse_args()



if __name__=='__main__':
    args = parse_args()
    feature_columns = ['time']
    if args.weather:
        feature_columns.extend(['temp','dew','precip','precipprob','visibility','solarradiation','severerisk',
            'preciptype','winddir','windgust','windspeed'])

    if args.scale:
        feature_columns.extend(['scale_sum_r_o','scale_max_r_o', 'scale_std_r_o'])
    feature_length = len(feature_columns)
    print('feature_columns:',feature_length)
    batch_size = args.batch_size
    unfreeze_epoch = args.unfreez_epoch

    print('Model variant:', args.variant)


    if args.variant=='b0':
        save_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pth"
        plot_path = f'{project_root_dir}/results/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.png'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pkl'

        folder = f"{project_root_dir}/models/segformer_training/work_dirs/segformer_512_sbatch"
    else:
        save_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_attention.pth"
        plot_path = f'{project_root_dir}/results/fold_seg_reg_{args.fold}_{feature_length}_attention.png'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_attention.pkl'
        folder = f"{project_root_dir}/models/segformer_training/work_dirs/segformer_512_sbatch"


    # List all files in the folder
    all_files = os.listdir(folder)

    # Filter for files starting with 'best_mIoU_iter_' and ending with '.pth'
    pth_files = [f for f in all_files if f.startswith("best_mIoU_iter_") and f.endswith(".pth")]

    if not pth_files:
        raise FileNotFoundError("No matching .pth file found.")

    # Sort based on iteration number
    pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Get full path of the latest checkpoint
    encoder_ckpt_path = os.path.join(folder, pth_files[-1])

    print("Segformer's Best checkpoint path:", encoder_ckpt_path)




    model = SegFormerRegressor(
            variant = args.variant,
            encoder_ckpt=encoder_ckpt_path,
            num_extra_feats=len(feature_columns),
            freeze_encoder=True
        )


    train_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/train_split_{args.fold}.csv")
    val_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/val_split_{args.fold}.csv")
    test_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/test_split.csv")

    print('#train:',len(train_df), '#val:',len(val_df), '#test:',len(test_df))

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])

    # Save this for inference later
    joblib.dump(scaler, save_scalers_of_features)


    image_folders = [f'{data_base_path}/Dataset_images_2024', f'{data_base_path}/Dataset_images_2025']
    image_size = 768
    train_transform = A.Compose([
        # 1. Resize longest side to target size while keeping aspect ratio
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        
        # 2. Pad the shorter side with zeros to make it a square (image_size x image_size)
        A.PadIfNeeded(
            min_height=image_size, 
            min_width=image_size, 
            border_mode=0,  # 0 = constant padding (black)
            value=(0, 0, 0)
        ),
        
        # Optional: If you still want a cropping augmentation that respects aspect ratio,
        # you can use RandomCrop instead of RandomResizedCrop after padding, or stick to flips/jitters:
        A.HorizontalFlip(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
        
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        # 1. Resize longest side while keeping aspect ratio
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        
        # 2. Pad to make it perfectly square without distortion
        A.PadIfNeeded(
            min_height=image_size, 
            min_width=image_size, 
            border_mode=0, 
            value=(0, 0, 0)
        ),
        
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    # Rebuild datasets using saved splits and scaler
    train_dataset = FruitDataset(f"{project_root_dir}/data/train_test_splits/train_split_{args.fold}.csv", image_folders, feature_columns, transform=train_transform, scaler=scaler)
    val_dataset   = FruitDataset(f"{project_root_dir}/data/train_test_splits/val_split_{args.fold}.csv", image_folders, feature_columns, transform=val_test_transform, scaler=scaler)
    test_dataset  = FruitDataset(f"{project_root_dir}/data/train_test_splits/test_split.csv", image_folders, feature_columns, transform=val_test_transform, scaler=scaler)

    # And create new dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)



    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"Current device: {current_device} - {torch.cuda.get_device_name(current_device)}")
            print("Devices visible to PyTorch:")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        model = torch.nn.DataParallel(model)

    model = model.to(device)  # wrap first, then move to CUDA
    # Loss and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # optimizer = your optimizer, e.g., Adam
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Training config
    num_epochs = 100
    best_val_loss = float('inf')
    early_stop = 0
    # early_stop_t = 10
    early_stop_t = 30

    train_losses=[]
    val_losses = []
    train_maes, val_maes = [], []
    train_r2s, val_r2s = [], []
    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            print('unfreezing the encoder')
            if hasattr(model, 'module'):
                model.module.unfreeze_encoder()
            else:
                model.unfreeze_encoder()
            # model.module.unfreeze_encoder()  # <--- this line unfreezes

        model.train()
        train_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        for images, times, targets in tqdm(train_loader):
            # print('images:', images.shape, 'times', times.shape, 'targets:',targets.shape )
            images = images.to(device)
            # times = times.to(device)
            times = times.to(device).float()  # must be float for Linear layer

            targets = targets.to(device).unsqueeze(1)
            if len(times.shape)==1:
                times= times.to(device).unsqueeze(1)
            # print('images:',images.shape, 'times:',times.shape)
            preds = model(images, times)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_preds_list.append(preds.detach().cpu())
            train_targets_list.append(targets.detach().cpu())
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Concatenate all predictions and targets for MAE and R²
        train_preds_all = torch.cat(train_preds_list).numpy()
        train_targets_all = torch.cat(train_targets_list).numpy()

        train_mae = mean_absolute_error(train_targets_all, train_preds_all)
        train_r2 = r2_score(train_targets_all, train_preds_all)

        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        # Validation
        val_preds_list = []
        val_targets_list = []
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, times, targets in tqdm(val_loader):
                images = images.to(device)
                # times = times.to(device)
                times = times.to(device).float()  # must be float for Linear layer

                targets = targets.to(device).unsqueeze(1)
                if len(times.shape)==1:
                    times= times.to(device).unsqueeze(1)
                preds = model(images, times)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
                val_preds_list.append(preds.cpu())
                val_targets_list.append(targets.cpu())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_preds_all = torch.cat(val_preds_list).numpy()
        val_targets_all = torch.cat(val_targets_list).numpy()

        val_mae = mean_absolute_error(val_targets_all, val_preds_all)
        val_r2 = r2_score(val_targets_all, val_preds_all)
        scheduler.step(val_mae)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print('Best model saved!!')
            early_stop = 0
        else:
            early_stop +=1 
            print('early_stop:', early_stop)
        if early_stop==early_stop_t:
            print('early_stop triggered!!')
            break

        print(f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train MSE: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f} | "
        f"Val MSE: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}"
        )
        print(f"lr:{optimizer.param_groups[0]['lr']}")

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_maes=train_maes,
        val_maes=val_maes,
        train_r2s=train_r2s,
        val_r2s=val_r2s,
        save_path=plot_path
    )

    print('Model Testing...........................................................................')
    def remove_module_prefix(state_dict):
        """Removes 'module.' prefix from keys in a state_dict"""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        return new_state_dict

    # Load checkpoint
    checkpoint = torch.load(save_path, map_location='cpu')

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefixes
    cleaned_state_dict = remove_module_prefix(state_dict)

    # Load into model
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print("✅ Loaded checkpoint with cleaned keys.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Move model to device
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)  # wrap first, then move to CUDA
    # Loss and optimizer
    criterion = nn.MSELoss()

    # TEST
    test_preds_list = []
    test_targets_list = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, times, targets in tqdm(test_loader):
            images = images.to(device)
            # times = times.to(device)
            times = times.to(device).float()  # must be float for Linear layer

            targets = targets.to(device).unsqueeze(1)
            if len(times.shape)==1:
                times= times.to(device).unsqueeze(1)
            preds = model(images, times)
            loss = criterion(preds, targets)
            test_loss += loss.item() * images.size(0)
            test_preds_list.append(preds.cpu())
            test_targets_list.append(targets.cpu())

    test_loss /= len(test_loader.dataset)

    test_preds_all = torch.cat(test_preds_list).numpy()
    test_targets_all = torch.cat(test_targets_list).numpy()

    test_mae = mean_absolute_error(test_targets_all, test_preds_all)
    test_r2 = r2_score(test_targets_all, test_preds_all)

    print(f"Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    # python segmango_ssh/models/approach_2/train_segmango.py --weather --scale --fold 1 --variant b1 --unfreez_epoch 30
