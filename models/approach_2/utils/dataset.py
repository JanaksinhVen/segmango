import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FruitDataset(Dataset):
    def __init__(self, csv_path, image_folders, feature_columns, transform=None, scaler=None, image_exts=['.jpg', '.png']):
        self.data = pd.read_csv(csv_path)
        self.image_folders = image_folders
        self.feature_columns = feature_columns
        self.transform = transform
        self.image_exts = image_exts
        self.scaler = scaler  # 💡 Save scaler

    def find_image_path(self, image_name):
        for folder in self.image_folders:
            for ext in self.image_exts:
                path = os.path.join(folder, image_name + ext)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(f"Image {image_name} not found in folders {self.image_folders} with extensions {self.image_exts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image_name']
        image_path = self.find_image_path(image_name)
        image = Image.open(image_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        image = np.array(image)  # ✅ convert PIL to NumPy
        if self.transform:
            image = self.transform(image=image)['image']


        # Tabular features
        feature_df = pd.DataFrame([row[self.feature_columns].values], columns=self.feature_columns)
        # features = row[self.feature_columns].values.astype('float32').reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(feature_df)  # 💡 Normalize
        features = torch.tensor(features.squeeze(), dtype=torch.float32)

        # Regression target
        target = torch.tensor(float(row['n_fruit_o']), dtype=torch.float32)

        return image, features, target


class FruitDataset_per_tree(Dataset):
    def __init__(self, csv_path,list_of_trees, manual_csv, image_folders, feature_columns, transform=None, scaler=None, image_exts=['.jpg', '.png']):
        self.data = pd.read_csv(csv_path)
        self.manual_csv = pd.read_csv(manual_csv)
        self.image_folders = image_folders
        self.feature_columns = feature_columns
        self.transform = transform
        self.image_exts = image_exts
        self.scaler = scaler  # 💡 Save scaler
        self.list_of_trees = list_of_trees

    def find_image_path(self, image_name):
        for folder in self.image_folders:
            for ext in self.image_exts:
                path = os.path.join(folder, image_name + ext)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(f"Image {image_name} not found in folders {self.image_folders} with extensions {self.image_exts}")

    def __len__(self):
        return len(self.list_of_trees)

    def image_groups(self,idx):
        # print('idx:',idx)
        tree_ = self.list_of_trees[idx]
        img_names = [tree_+f'_0{i+1}' for i in range(8)]
        return [self.find_image_path(img_name) for img_name in img_names]

    def __getitem__(self, idx):
        images = [Image.open(p).convert('RGB') for p in self.image_groups(idx)]
        if self.transform:
            images = [self.transform(image=np.array(img))['image'] for img in images]

        images = torch.stack(images)  # (8, 3, H, W)

        # row = self.data.iloc[idx]
        # image_name = row['image_name']
        # image_path = self.find_image_path(image_name)
        # image = Image.open(image_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        tree_ = self.list_of_trees[idx]
        img_names = [tree_+f'_0{i+1}' for i in range(8)]
        feature_df = self.data[self.data['image_name'].isin(img_names)][self.feature_columns]
        
        # targets_ = self.data[self.data['image_name'].isin(img_names)]['n_fruit_o']
        if self.scaler is not None:
            features = self.scaler.transform(feature_df)  # 💡 Normalize



        if not tree_.startswith('N_'):
            row_n = tree_.split('_')[-1]
            targets_ = self.manual_csv[self.manual_csv['Tree_number'] == row_n]['count'].values[0]

        else:
            row_n = 'N_' + tree_.split('_')[-1]
            targets_ =  self.manual_csv[self.manual_csv['Tree_number'] == row_n]['count'].values[0]

        # # Tabular features
        # feature_df = pd.DataFrame([row[self.feature_columns].values], columns=self.feature_columns)
        # # features = row[self.feature_columns].values.astype('float32').reshape(1, -1)
        # if self.scaler is not None:
        #     features = self.scaler.transform(feature_df)  # 💡 Normalize
        # features = torch.tensor(features.squeeze(), dtype=torch.float32)

        # Regression target
        # target = torch.tensor(float(row['n_fruit_o']), dtype=torch.float32)
        targets = torch.tensor(targets_.astype(np.float32), dtype=torch.float32)

        return images, features, targets
