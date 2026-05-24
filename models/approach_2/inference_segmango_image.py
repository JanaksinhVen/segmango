
import argparse
import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
project_root_dir = os.getenv("ROOT_DIR")

# Import your custom architecture
from utils.segmango_model import SegFormerRegressor

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for SegFormer Regressor")
    
    # Required Inputs
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--time', type=float, default=0.0, help='Time metadata feature value')
    parser.add_argument('--fold', type=int, required=True, help='Fold number of the model to use')
    parser.add_argument('--variant', type=str, required=True, help='Model variant (e.g., b0, b1)')
    
    # Feature Flags (Matches training logic)
    parser.add_argument('--weather', action='store_true', help='Include weather features')
    parser.add_argument('--scale', action='store_true', help='Include scale features')
    
    # Optional Weather Features (Only used if --weather is provided)
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--dew', type=float, default=0.0)
    parser.add_argument('--precip', type=float, default=0.0)
    parser.add_argument('--precipprob', type=float, default=0.0)
    parser.add_argument('--visibility', type=float, default=0.0)
    parser.add_argument('--solarradiation', type=float, default=0.0)
    parser.add_argument('--severerisk', type=float, default=0.0)
    parser.add_argument('--preciptype', type=float, default=0.0) # Ensure matches your training encoding numeric type
    parser.add_argument('--winddir', type=float, default=0.0)
    parser.add_argument('--windgust', type=float, default=0.0)
    parser.add_argument('--windspeed', type=float, default=0.0)
    
    # Optional Scale Features (Only used if --scale is provided)
    parser.add_argument('--scale_sum_r_o', type=float, default=0.0)
    parser.add_argument('--scale_max_r_o', type=float, default=0.0)
    parser.add_argument('--scale_std_r_o', type=float, default=0.0)

    return parser.parse_args()


def remove_module_prefix(state_dict):
    """Removes 'module.' prefix from keys if model was trained with DataParallel"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def main():
    args = parse_args()
    
    # 1. Reconstruct feature column list exactly like training
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
    print(f"ℹ️ Configured feature columns count: {feature_length}")

    # 2. Build paths to the saved model weights and matching scalers
    if args.variant == 'b0':
        weight_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pth"
        scaler_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_{args.variant}_attention.pkl"
    else:
        weight_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/fold_seg_reg_{args.fold}_{feature_length}_attention.pth"
        scaler_path = f"{project_root_dir}/data/Model_weights/approach-1/segmango/S_fold_seg_reg_{args.fold}_{feature_length}_attention.pkl"

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ Weight file not found at: {weight_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler file not found at: {scaler_path}")

    # 3. Process Tabular Metadata Features
    # Create a dictionary mapping keys to provided arguments
    input_meta_dict = {feat: getattr(args, feat) for feat in feature_columns}
    df_meta = pd.DataFrame([input_meta_dict])
    
    # Scale features using training scaler
    scaler = joblib.load(scaler_path)
    scaled_meta = scaler.transform(df_meta[feature_columns])
    meta_tensor = torch.tensor(scaled_meta, dtype=torch.float32)

    # 4. Process Image File
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"❌ Image file not found at: {args.image_path}")
        
    image = np.array(Image.open(args.image_path).convert("RGB"))
    
    # Image pipeline mirroring val_test_transform
    image_size = 768
    inference_transform = A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=(0, 0, 0)),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = inference_transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0) # Add batch dimension (1, C, H, W)

    # 5. Initialize Model Architecture & Load Weights
    print(f"🔄 Initializing SegFormerRegressor variant '{args.variant}'...")
    model = SegFormerRegressor(
        variant=args.variant,
        encoder_ckpt=None, # Not required for purely executing inference weights
        num_extra_feats=feature_length,
        freeze_encoder=True
    )
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    cleaned_state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 6. Execute Model Prediction
    image_tensor = image_tensor.to(device)
    meta_tensor = meta_tensor.to(device)

    print("🚀 Running prediction...")
    with torch.no_grad():
        prediction = model(image_tensor, meta_tensor)
        output_value = prediction.cpu().item()

    print(f"\n🎯 Prediction Result: {output_value:.4f}")
    print("final mango count:", round(output_value))

if __name__ == '__main__':
    main()

# python segmango_ssh/models/approach_2/inference_segmango_image.py --image "segmango_ssh/02_10_02.jpg" --fold 1 --variant b1 --time 64
