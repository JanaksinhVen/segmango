import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

# Get the ROOT_DIR variable
data_base_path = os.getenv("DATA_DIR")
project_root_dir = os.getenv("ROOT_DIR")
# 1. Setup Directories
results_dir = os.path.join(project_root_dir, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 2. Configuration
fold_id = 1
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 3. Load Data
df_test = pd.read_csv(f'{project_root_dir}/data/train_test_splits/test_split.csv')
df_train = pd.read_csv(f'{project_root_dir}/data/train_test_splits/train_split_{fold_id}.csv')
df_val = pd.read_csv(f'{project_root_dir}/data/train_test_splits/val_split_{fold_id}.csv')

# Concatenate for CV
data = pd.concat([df_train, df_val]).reset_index(drop=True)

features = [
    'n_flower', 'avg_flower', 'n_fruit', 'avg_fruit', 'time',
    'temp', 'dew', 'precip', 'visibility', 'solarradiation',
    'severerisk', 'winddir', 'windgust', 'windspeed',
    'scale_sum_r_o', 'scale_max_r_o', 'scale_std_r_o'
]

# Prepare Features and Target
X = data[features].values
Y = data[['n_fruit_o']].values
X_test = df_test[features].values
Y_test = df_test[['n_fruit_o']].values

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors (as per your original structure)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# 5. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

fold_results = []
fold_test_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_train_np, Y_train_np = X_tensor[train_idx].numpy(), Y_tensor[train_idx].numpy().ravel()
    X_val_np, Y_val_np = X_tensor[val_idx].numpy(), Y_tensor[val_idx].numpy().ravel()
    X_test_np, Y_test_np = X_test_tensor.numpy(), Y_test_tensor.numpy().ravel()

    # Model
    model = LinearRegression()
    model.fit(X_train_np, Y_train_np)
    
    # Validation Eval
    Y_val_pred = model.predict(X_val_np)
    r2_v = r2_score(Y_val_np, Y_val_pred)
    mse_v = mean_squared_error(Y_val_np, Y_val_pred)
    mae_v = mean_absolute_error(Y_val_np, Y_val_pred)
    
    fold_results.append({'fold': fold+1, 'r2': r2_v, 'mse': mse_v, 'mae': mae_v})

    # Test Eval
    Y_test_pred = model.predict(X_test_np)
    r2_t = r2_score(Y_test_np, Y_test_pred)
    mse_t = mean_squared_error(Y_test_np, Y_test_pred)
    mae_t = mean_absolute_error(Y_test_np, Y_test_pred)
    
    fold_test_results.append({'fold': fold+1, 'r2': r2_t, 'mse': mse_t, 'mae': mae_t})

# 6. Summarize and Save Results
res_df_val = pd.DataFrame(fold_results)
res_df_test = pd.DataFrame(fold_test_results)

# Calculate Statistics
stats = {
    "Metric": ["R2", "MSE", "MAE"],
    "Val_Mean": [res_df_val['r2'].mean(), res_df_val['mse'].mean(), res_df_val['mae'].mean()],
    "Val_Std": [res_df_val['r2'].std(), res_df_val['mse'].std(), res_df_val['mae'].std()],
    "Test_Mean": [res_df_test['r2'].mean(), res_df_test['mse'].mean(), res_df_test['mae'].mean()],
    "Test_Std": [res_df_test['r2'].std(), res_df_test['mse'].std(), res_df_test['mae'].std()]
}
stats_df = pd.DataFrame(stats)

# Save to CSV
res_df_val.to_csv(os.path.join(results_dir, f'approach_1_val_fold_results_{fold_id}.csv'), index=False)
res_df_test.to_csv(os.path.join(results_dir, f'approach_1_test_fold_results_{fold_id}.csv'), index=False)
stats_df.to_csv(os.path.join(results_dir, f'approach_1_summary_stats_{fold_id}.csv'), index=False)

# Console Printout
print("\n" + "="*30)
print("Approach 1, RESULTS SUMMARY")
print("="*30)
print(f"Validation R²: {stats['Val_Mean'][0]:.4f} ± {stats['Val_Std'][0]:.4f}")
print(f"Test R²:       {stats['Test_Mean'][0]:.4f} ± {stats['Test_Std'][0]:.4f}")
print(f"\nFiles saved to: {results_dir}")


# python approach_1.py