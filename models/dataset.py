"""
Dataset classes and data loading utilities.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FruitDataset(Dataset):
    """
    Custom dataset for fruit count regression with image and tabular features.
    
    Args:
        csv_path: Path to CSV file with image names and features
        image_folders: List of folders containing images
        feature_columns: List of feature column names
        transform: Albumentations transform pipeline
        scaler: StandardScaler fitted on training data
        image_exts: List of valid image extensions
    """
    
    def __init__(self, csv_path, image_folders, feature_columns, 
                 transform=None, scaler=None, image_exts=['.jpg', '.png']):
        self.data = pd.read_csv(csv_path)
        self.image_folders = image_folders
        self.feature_columns = feature_columns
        self.transform = transform
        self.image_exts = image_exts
        self.scaler = scaler

    def find_image_path(self, image_name):
        """Find image path by searching all folders and extensions."""
        for folder in self.image_folders:
            for ext in self.image_exts:
                path = os.path.join(folder, image_name + ext)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(
            f"Image {image_name} not found in folders {self.image_folders} "
            f"with extensions {self.image_exts}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image_name']
        image_path = self.find_image_path(image_name)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']

        # Process tabular features
        feature_df = pd.DataFrame(
            [row[self.feature_columns].values], 
            columns=self.feature_columns
        )
        if self.scaler is not None:
            features = self.scaler.transform(feature_df)
        else:
            features = feature_df.values
        features = torch.tensor(features.squeeze(), dtype=torch.float32)

        # Target variable
        target = torch.tensor(float(row['n_fruit_o']), dtype=torch.float32)

        return image, features, target


def get_data_transforms(image_size=768):
    """
    Create training and validation/test augmentation pipelines.
    
    Args:
        image_size: Target image size (H, W)
        
    Returns:
        Tuple of (train_transform, val_test_transform)
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(
            height=image_size, 
            width=image_size, 
            scale=(0.8, 1.0), 
            interpolation=1
        ),
        A.HorizontalFlip(p=0.3),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.1, 
            hue=0.02, 
            p=0.5
        ),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    val_test_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=1),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    return train_transform, val_test_transform


def get_dataframes_from_splits(full_df, train_file, val_file, test_file):
    """
    Load train/val/test dataframes from split text files.
    
    Args:
        full_df: Full dataframe with all samples
        train_file: Path to train split file
        val_file: Path to validation split file
        test_file: Path to test split file
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    def get_df(df, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = lines[i].split('\n')[0]
            if line.split('_')[0] + '_' + line.split('_')[1] == 'N_03':
                lines[i] = lines[i].replace('N_03', 'N_01')
        return df[df['image_name'].isin(lines)]
    
    train_df = get_df(full_df, train_file)
    val_df = get_df(full_df, val_file)
    test_df = get_df(full_df, test_file)
    
    return train_df, val_df, test_df
