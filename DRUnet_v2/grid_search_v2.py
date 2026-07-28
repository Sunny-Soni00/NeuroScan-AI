import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset_balance_v2 import BraTSDataset25D
from model_drunet_v2 import AttentionDRUNet
from utils_v2 import HybridFocalDiceLoss, calculate_all_metrics
from tqdm import tqdm
import numpy as np

# ================= ⚙️ PERFORMANCE CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 4
SUBSET_MODE = True
SUBSET_RATIO = 0.05
RANDOM_SEED = 42  # For scientific reproducibility
DATA_BASE_PATH = "../BraTS_Split"
RESULTS_DIR = "results/grid_search"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Grid values
lambda_values = [0.3, 0.5, 0.7]
gamma_values = [1.5, 2.0, 2.5]

def run_grid_search():
    csv_path = os.path.join(RESULTS_DIR, "grid_search_results.csv")
    
    # 🔄 Load existing results to resume if interrupted
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        completed_combos = set(zip(results_df['lambda'], results_df['gamma']))
        results = results_df.to_dict('records')
        print(f"🔄 Resuming: Found {len(completed_combos)} completed combinations.")
    else:
        results = []
        completed_combos = set()

    # Initialize Dataset
    np.random.seed(RANDOM_SEED)
    full_train_ds = BraTSDataset25D(os.path.join(DATA_BASE_PATH, "train"))
    val_ds = BraTSDataset25D(os.path.join(DATA_BASE_PATH, "val"))

    if SUBSET_MODE:
        indices = np.random.choice(len(full_train_ds), int(len(full_train_ds) * SUBSET_RATIO), replace=False)
        train_ds = Subset(full_train_ds, indices)
        print(f"📊 Subset Mode: Using {len(train_ds)} slices ({SUBSET_RATIO*100}%).")
    else:
        train_ds = full_train_ds

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    scaler = torch.amp.GradScaler('cuda')

    for l_val in lambda_values:
        for g_val in gamma_values:
            # Skip if already done
            if (l_val, g_val) in completed_combos:
                continue

            print(f"\n🚀 Testing: Lambda={l_val}, Gamma={g_val}")
            model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = HybridFocalDiceLoss(gamma=g_val, alpha=l_val) 
            
            # Training Trend (2 Epochs)
            model.train()
            for epoch in range(2):
                loop = tqdm(train_loader, desc=f"L={l_val} G={g_val} Ep {epoch+1}")
                for x, y in loop:
                    x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                    with torch.amp.autocast('cuda'):
                        preds = model(x)
                        loss = loss_fn(preds, y)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loop.set_postfix(loss=loss.item())

            # Validation
            model.eval()
            val_metrics = {"dice": 0, "precision": 0, "recall": 0}
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                    with torch.amp.autocast('cuda'):
                        preds = model(x)
                    m = calculate_all_metrics(preds, y)
                    for k in val_metrics: val_metrics[k] += m[k]
            
            # Calculate Averages
            entry = {
                "lambda": l_val, 
                "gamma": g_val, 
                "dice": val_metrics["dice"] / len(val_loader),
                "precision": val_metrics["precision"] / len(val_loader),
                "recall": val_metrics["recall"] / len(val_loader)
            }
            results.append(entry)
            
            # 💾 Save Point: Update CSV after every combination
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"✅ Saved combo L={l_val}, G={g_val} (Dice: {entry['dice']:.4f})")

    # Final Plotting Logic
    print("\n📊 Generating Final Heatmap...")
    df = pd.read_csv(csv_path)
    pivot_df = df.pivot(index="lambda", columns="gamma", values="dice")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Hyperparameter Sensitivity Heatmap")
    plt.savefig(os.path.join(RESULTS_DIR, "grid_search_heatmap.png"))
    plt.close()
    print(f"📂 Process Complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    run_grid_search()