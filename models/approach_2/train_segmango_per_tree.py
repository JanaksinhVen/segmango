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

from utils.dataset import FruitDataset_per_tree
from utils.segmango_model import SegFormerRegressor
from utils.segmango_model import MultiImageSegFormerRegressor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    # parser.add_argument('--tree_n', type=int, required=True, help='tree number for iso-tree validation')
    # parser.add_argument('--new_tree', action='store_true', help='Include weather features')

    parser.add_argument('--weather', action='store_true', help='Include weather features')
    parser.add_argument('--scale', action='store_true', help='Include scale features')
    parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
    parser.add_argument('--variant', type=str, required=True, help='Model varient')
    parser.add_argument('--unfreez_epoch', type=int, default=30, help='Epoch to start unfreezing the encoder')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')

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
    # print('tree number:',args.tree_n,args.new_tree)
    # feature_columns = ['time',
    # 'scale_sum_r', 'scale_max_r', 'scale_std_r',
    #         'temp','dew','precip','precipprob','visibility','solarradiation','severerisk',
    #         'preciptype','winddir','windgust','windspeed']

    batch_size = args.batch_size
    unfreeze_epoch = args.unfreez_epoch
    # save_path = f"/home2/janakv/yield_pred/code/checkpoints/folds/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}.pth"
    # plot_path = f'/home2/janakv/yield_pred/code/seg_reg_resutls/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}.png'

    # save_scalers_of_features = f'/home2/janakv/yield_pred/code/checkpoints/S_fold_seg_reg_{args.fold}_{feature_length}_{args.variant}.pkl'
    print('Model variant:', args.variant)




    # encoder_ckpt_path = "/home2/janakv/yield_pred/work_dirs/segformer_768_sbatch_1/best_mIoU_iter_23000.pth"

    if args.variant=='b0':
        save_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pth"
        save_path_PT = f'{project_root_dir}/data/Model_weights/approach-1/segmango/best_model_per_tree_{args.fold}_{feature_length}_{args.variant}_attention.pth'
        save_plot_ = f'{project_root_dir}/results/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.png'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pkl'

        folder = f"{project_root_dir}/models/segformer_training/work_dirs/segformer_512_sbatch"
        # folder = f"{project_root_dir}/yield_pred/work_dirs/segformer_768_sbatch_1_fold_{args.fold}_{args.variant}"
    else:
        save_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_attention.pth"
        save_path_PT = f'{project_root_dir}/data/Model_weights/approach-1/segmango/best_model_per_tree_{args.fold}_{feature_length}_attention.pth'
        save_plot_ = f'{project_root_dir}/results/fold_seg_reg_{args.fold}_{feature_length}_attention.png'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_attention.pkl'
        # folder = f"/home2/janakv/yield_pred/work_dirs/segformer_768_sbatch_1_fold_{args.fold}"
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
    print('Loading per image model from path: ',save_path)

    manual_tree_count = f"{project_root_dir}/data/mango_manual_count_2024_2025.csv"

    def list_of_tree(df):
        lines = df['image_name']
        # with open(file_path,'r') as f:
        #     lines = f.readlines()
        # for i,line in enumerate(lines):
        #     lines[i]=lines[i].split('\n')[0]
        #     if line.split('_')[0]+'_'+line.split('_')[1]=='N_03':
        #         lines[i]=lines[i].replace('N_03','N_01')
        print(len(lines))
        return list({ '_'.join(f.split('_')[:-1]) for f in lines})



    train_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/train_split_{args.fold}.csv")
    val_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/val_split_{args.fold}.csv")
    test_df = pd.read_csv(f"{project_root_dir}/data/train_test_splits/test_split.csv")

    print('#train:',len(train_df), '#val:',len(val_df), '#test:',len(test_df))
    list_of_trees_train = list_of_tree(train_df)
    list_of_trees_val = list_of_tree(val_df)
    list_of_trees_test = list_of_tree(test_df)

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
    train_dataset = FruitDataset_per_tree(f"{project_root_dir}/data/train_test_splits/train_split_{args.fold}.csv",list_of_trees_train, manual_tree_count, image_folders, feature_columns, transform=train_transform, scaler=scaler)
    val_dataset   = FruitDataset_per_tree(f"{project_root_dir}/data/train_test_splits/val_split_{args.fold}.csv", list_of_trees_val, manual_tree_count, image_folders, feature_columns, transform=val_test_transform, scaler=scaler)
    test_dataset  = FruitDataset_per_tree(f"{project_root_dir}/data/train_test_splits/test_split.csv", list_of_trees_test, manual_tree_count, image_folders, feature_columns, transform=val_test_transform, scaler=scaler)

    # And create new dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    seg_reg_model = SegFormerRegressor(
            variant = args.variant,
            encoder_ckpt=encoder_ckpt_path,
            num_extra_feats=len(feature_columns),
            freeze_encoder=True,
            freeze_regressor=True
        )

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
    missing, unexpected = seg_reg_model.load_state_dict(cleaned_state_dict, strict=False)

    print("✅ Loaded checkpoint with cleaned keys.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # for image, features, targets in tqdm(train_loader):
    #     print(image.shape, features.shape, targets.shape)
    #     break

    model = MultiImageSegFormerRegressor(base_model=seg_reg_model).to(device)


    # Setup
    # unfreeze_epoch = 50
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    model.to(device)
    num_epochs = 100
    best_val_loss = float('inf')

    # Metric trackers
    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []
    train_maes, val_maes = [], []
    train_mses, val_mses = [], []
    early_stop = 0
    # early_stop_t = 10
    early_stop_t = 30

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            print('unfreezing the Regressor')
            if hasattr(model, 'module'):
                model.module.base_model.unfreeze_regressor()
            else:
                model.base_model.unfreeze_regressor()
            # model.module.unfreeze_encoder()  # <--- this line unfreezes

        model.train()
        epoch_train_preds, epoch_train_targets = [], []
        epoch_loss = 0.0

        for imgs, feats, target in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            imgs = imgs.to(device)
            feats = feats.to(device) if feats is not None else None
            target = target.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(imgs, extra_feats=feats)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            epoch_train_preds.extend(output.detach().cpu().numpy())
            epoch_train_targets.extend(target.cpu().numpy())

        # Train metrics
        train_loss = epoch_loss / len(train_loader.dataset)
        train_preds = torch.tensor(epoch_train_preds)
        train_targets = torch.tensor(epoch_train_targets)
        train_r2 = r2_score(train_targets, train_preds)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_mse = mean_squared_error(train_targets, train_preds)

        train_losses.append(train_loss)
        train_r2s.append(train_r2)
        train_maes.append(train_mae)
        train_mses.append(train_mse)

        print(f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}")

        # ============== Validation ==============
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for imgs, feats, target in tqdm(val_loader, desc=f"[Train] Epoch {epoch+1}"):
                imgs = imgs.to(device)
                feats = feats.to(device) if feats is not None else None
                target = target.to(device).unsqueeze(1)

                output = model(imgs, extra_feats=feats)
                loss = criterion(output, target)

                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds = torch.tensor(val_preds)
        val_targets = torch.tensor(val_targets)
        val_r2 = r2_score(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_mse = mean_squared_error(val_targets, val_preds)
        scheduler.step(val_mae)

        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        val_maes.append(val_mae)
        val_mses.append(val_mse)

        print(f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path_PT)
            print(f"[INFO] Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
            early_stop = 0
        else:
            early_stop +=1 
            print('early_stop:', early_stop)
        if early_stop==early_stop_t:
            print('early_stop triggered!!')
            break

    # epochs = range(1, num_epochs + 1)

    # def plot_metric(train_values, val_values, metric_name):
    #     plt.plot(epochs, train_values, 'b-', label=f"Train {metric_name}")
    #     plt.plot(epochs, val_values, 'r-', label=f"Val {metric_name}")
    #     plt.xlabel("Epochs")
    #     plt.ylabel(metric_name)
    #     plt.title(f"{metric_name} Over Epochs")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # # Plot all metrics
    # plot_metric(train_losses, val_losses, "Loss")
    # plot_metric(train_r2s, val_r2s, "R² Score")
    # plot_metric(train_maes, val_maes, "MAE")
    # plot_metric(train_mses, val_mses, "MSE")


    epochs = range(1, len(train_losses) + 1)

    # Create figure and 2×2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training & Validation Metrics Over Epochs", fontsize=16)

    # Plot Loss
    axs[0, 0].plot(epochs, train_losses, 'b-', label="Train Loss")
    axs[0, 0].plot(epochs, val_losses, 'r-', label="Val Loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot R²
    axs[0, 1].plot(epochs, train_r2s, 'b-', label="Train R²")
    axs[0, 1].plot(epochs, val_r2s, 'r-', label="Val R²")
    axs[0, 1].set_title("R² Score")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("R²")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot MAE
    axs[1, 0].plot(epochs, train_maes, 'b-', label="Train MAE")
    axs[1, 0].plot(epochs, val_maes, 'r-', label="Val MAE")
    axs[1, 0].set_title("MAE")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("MAE")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot MSE
    axs[1, 1].plot(epochs, train_mses, 'b-', label="Train MSE")
    axs[1, 1].plot(epochs, val_mses, 'r-', label="Val MSE")
    axs[1, 1].set_title("MSE")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("MSE")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_plot_, dpi=300)
    plt.show()


    # load_path = '/home2/janakv/yield_pred/code/checkpoints/best_model_overall_per_tree_1.pth'
    load_path=save_path_PT
    def remove_module_prefix(state_dict):
        """Removes 'module.' prefix from keys in a state_dict"""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        return new_state_dict

    # Load checkpoint
    checkpoint = torch.load(load_path, map_location='cpu')

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefixes
    cleaned_state_dict = remove_module_prefix(state_dict)

    # Load into model
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print("✅ Loaded checkpoint with cleaned keys for test")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    test_loss, test_preds, test_targets = 0.0, [], []
    model.to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        for imgs, feats, target in tqdm(test_loader, desc=f"[Train] Epoch {epoch+1}"):
            imgs = imgs.to(device)
            feats = feats.to(device) if feats is not None else None
            target = target.to(device).unsqueeze(1)

            output = model(imgs, extra_feats=feats)
            loss = criterion(output, target)

            test_loss += loss.item() * imgs.size(0)
            test_preds.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_preds = torch.tensor(test_preds)
    test_targets = torch.tensor(test_targets)
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_mse = mean_squared_error(test_targets, test_preds)


    print(f"Test Loss: {test_loss:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")


# python segmango_ssh/models/approach_2/train_segmango_per_tree.py --weather --scale --fold 1 --variant b1 --unfreez_epoch 50
