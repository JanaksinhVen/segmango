import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import joblib
from dotenv import load_dotenv

# Load workspace paths from environment variables
load_dotenv()
project_root_dir = os.getenv("ROOT_DIR")

# Import custom architecture definitions
from utils.segmango_model import SegFormerRegressor, MultiImageSegFormerRegressor

# Set computation backend
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Per-Tree Multi-Image Inference Script with Unique View Metadata")
    
    # Structural Inputs
    parser.add_argument('--image_prefix', type=str, required=True, 
                        help="Full path prefix to look up the 8 images (e.g. '/scratch/janakv/Dataset_images_2024/02_10_')")
    parser.add_argument('--csv_path', type=str, required=True, 
                        help="Path to the metadata .csv file containing the tree parameters")
    parser.add_argument('--fold', type=int, required=True, help="Fold tracking number to align checkpoints")
    parser.add_argument('--variant', type=str, required=True, help="Model base backbones (e.g., b0, b1)")
    
    # Structural Parameter Flags matching training state
    parser.add_argument('--weather', action='store_true', help="Include environmental climate trackers")
    parser.add_argument('--scale', action='store_true', help="Include physical standard scale vectors")
    
    return parser.parse_args()


def remove_module_prefix(state_dict):
    """Removes 'module.' wrapper keys safely if trained via DataParallel structures"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def main():
    args = parse_args()

    # 1. Reconstruct feature structures to resolve matching dimensions
    feature_columns = ['time']
    if args.weather:
        feature_columns.extend([
            'temp', 'dew', 'precip', 'precipprob', 'visibility', 
            'solarradiation', 'severerisk', 'preciptype', 'winddir', 
            'windgust', 'windspeed'
        ])
    if args.scale:
        feature_columns.extend(['scale_sum_r_o', 'scale_max_r_o', 'scale_std_r_o'])
        
    feature_length = len(feature_columns)
    
    # Enforce precise runtime guardrail check matching input dimensions: [1, 12, 15]
    if feature_length not in [1, 12, 15]:
        raise ValueError(f"❌ Configuration Matrix error: Evaluated feature dimension length count is {feature_length}. "
                         f"Only strict dimensions of 1, 12, or 15 are accepted by the network parameters.")

    # 2. Verify and gather the 8 image paths
    print(f"🔍 Searching files matching sequence mask context: {args.image_prefix}*.jpg")
    img_lookups = sorted(glob.glob(f"{args.image_prefix}*.jpg"))
    if not img_lookups:
        img_lookups = sorted(glob.glob(f"{args.image_prefix}*.jpeg"))

    if len(img_lookups) != 8:
        raise FileNotFoundError(f"❌ Execution Blocked: Per-Tree array requirement demands exactly 8 image layers. "
                                f"Discovered count: {len(img_lookups)} targets.")

    # 3. Read CSV and isolate unique metadata for each of the 8 views separately
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"❌ Metadata CSV table source path not found: {args.csv_path}")
    
    csv_df = pd.read_csv(args.csv_path)
    
    # Resolve file name weight path routing
    if args.variant == 'b0':
        save_path_PT = f'{project_root_dir}/data/Model_weights/approach-1/segmango/best_model_per_tree_{args.fold}_{feature_length}_{args.variant}_attention.pth'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pkl'
    else:
        save_path_PT = f'{project_root_dir}/data/Model_weights/approach-1/segmango/best_model_per_tree_{args.fold}_{feature_length}_attention.pth'
        save_scalers_of_features = f'{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_attention.pkl'

    scaler = joblib.load(save_scalers_of_features)
    
    # Pre-allocate containers to hold ordered image views and paired features
    processed_images = []
    processed_features = []
    
    image_size = 768
    val_test_transform = A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=(0, 0, 0)),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

    print("🔄 Processing individual views paired with distinct CSV metadata rows...")
    for img_path in img_lookups:
        # Extract individual view identity code (e.g., '02_10_01')
        filename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. Process Image
        img_matrix = np.array(Image.open(img_path).convert("RGB"))
        transformed_payload = val_test_transform(image=img_matrix)
        processed_images.append(transformed_payload['image'])
        

        matching_row = csv_df[csv_df['image_name_o'].astype(str) == filename_without_ext]
        
        if matching_row.empty:
            # Fallback strategy: look for rows ending with the same view suffix if prefix directory structure differs
            view_suffix = filename_without_ext.split('_')[-1] # e.g. '01'
            tree_base = os.path.basename(args.image_prefix).rstrip('_') # e.g. '02_10'
            
            # Extract out via alternate pattern combination strings
            fallback_rows = csv_df[csv_df['image_name_o'].astype(str).str.contains(f"_{view_suffix}")]
            matching_row = fallback_rows[fallback_rows['image_name_o'].astype(str).str.contains(tree_base.split('_')[0])]
            
            if matching_row.empty:
                raise KeyError(f"❌ Structural matching error: View identity token '{filename_without_ext}' has no companion record in the CSV file.")

        # Extract metrics sequence slice, transform it, and convert to numpy array
        scaled_row = scaler.transform(matching_row.iloc[[0]][feature_columns])[0]
        processed_features.append(scaled_row)

    # 4. Construct structural multidimensional input blocks
    # Stack images -> Shape: [8, C, H, W] -> Batch dimension unsqueeze -> [1, 8, C, H, W]
    imgs_tensor = torch.stack(processed_images, dim=0).unsqueeze(0).to(device)
    
    # Stack unique features -> Shape: [8, F] -> Batch dimension unsqueeze -> [1, 8, F]
    feats_tensor = torch.tensor(np.array(processed_features), dtype=torch.float32).unsqueeze(0).to(device)
    
    print(f"🔹 Configured image tensor sequence shape: {imgs_tensor.shape}")
    print(f"🔹 Configured matched tabular feature tensor shape: {feats_tensor.shape}")

    # 5. Initialize Model Architecture & Load Combined Target Matrix Layers
    seg_reg_base = SegFormerRegressor(
        variant=args.variant,
        encoder_ckpt=None, 
        num_extra_feats=feature_length,
        freeze_encoder=True,
        freeze_regressor=True
    )
    
    model = MultiImageSegFormerRegressor(base_model=seg_reg_base)
    
    if not os.path.exists(save_path_PT):
        raise FileNotFoundError(f"❌ Weights not found at: {save_path_PT}")
        
    checkpoint = torch.load(save_path_PT, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    cleaned_state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 6. Execute Forward Pass Inference
    print("🚀 Computing structural forward tracking passes...")
    with torch.no_grad():
        prediction_output = model(imgs_tensor, extra_feats=feats_tensor)
        final_yield_prediction = prediction_output.cpu().item()

    print("\n" + "="*50)
    print(f"🎯 Target Tree Absolute Yield Prediction Result: {final_yield_prediction:.4f}")
    print("="*50)
    print(f"📌 Rounded Final Mango Count Estimate: {round(final_yield_prediction)}")

if __name__ == '__main__':
    main()

# python inference_per_tree.py --image_prefix "/scratch/janakv/Dataset_images_2024/02_10_" --csv_path "/home2/pronoy.patra/Segmango_project/segmango_ssh/data/mlp_all_data_with_time_weather_scale_treewise_2024.csv" --fold 1 --variant b1 --weather --scale