import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Import existing modules
from dataset_balance_v2 import BraTSDataset25D
from model_drunet_v2 import AttentionDRUNet
from utils_v2 import calculate_all_metrics

# ================= ⚙️ CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_BASE_PATH = "../../BraTS_Split"
MODEL_PATH = "results/v2_checkpoint.pth.tar" # Tera current epoch 20 model
RESULTS_DIR = "results/final_audit"
os.makedirs(RESULTS_DIR, exist_ok=True)

def apply_tta(model, x):
    """
    Test Time Augmentation:
    1. Predict Standard
    2. Predict Horizontal Flip
    3. Predict Vertical Flip
    Average all predictions.
    """
    # 1. Standard Prediction
    pred_standard = model(x)
    
    # 2. Horizontal Flip
    x_hflip = torch.flip(x, dims=[3])
    pred_hflip = model(x_hflip)
    pred_hflip_back = torch.flip(pred_hflip, dims=[3])
    
    # 3. Vertical Flip
    x_vflip = torch.flip(x, dims=[2])
    pred_vflip = model(x_vflip)
    pred_vflip_back = torch.flip(pred_vflip, dims=[2])
    
    # Average predictions (Ensemble effect)
    return (pred_standard + pred_hflip_back + pred_vflip_back) / 3.0

def run_evaluation():
    print(f"🚀 Starting Final Evaluation with TTA on {DEVICE}...")
    
    # Load Model
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights from {MODEL_PATH}")
    else:
        print("❌ Model checkpoint not found!")
        return

    model.eval()
    
    # Load Validation Data
    val_ds = BraTSDataset25D(os.path.join(DATA_BASE_PATH, "val"))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    metrics_acc = {"dice": [], "iou": [], "precision": [], "recall": []}

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating with TTA"):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            
            # Apply TTA (The Magic Step)
            with torch.amp.autocast('cuda'):
                preds = apply_tta(model, x)
            
            # Calculate Metrics
            batch_metrics = calculate_all_metrics(preds, y)
            
            # Store batch averages
            metrics_acc["dice"].append(batch_metrics["dice"])
            metrics_acc["iou"].append(batch_metrics["iou"])
            metrics_acc["precision"].append(batch_metrics["precision"])
            metrics_acc["recall"].append(batch_metrics["recall"])

    # Final Report
    print("\n" + "="*40)
    print(f"🏆 FINAL DRUNET V2 PERFORMANCE (w/ TTA)")
    print("="*40)
    print(f"🔹 Dice Score: {np.mean(metrics_acc['dice'])*100:.2f}%")
    print(f"🔹 IoU Score : {np.mean(metrics_acc['iou'])*100:.2f}%")
    print(f"🔹 Precision : {np.mean(metrics_acc['precision'])*100:.2f}%")
    print(f"🔹 Recall    : {np.mean(metrics_acc['recall'])*100:.2f}%")
    print("="*40)

    # Save to file
    with open(f"{RESULTS_DIR}/final_report.txt", "w") as f:
        f.write(f"Dice: {np.mean(metrics_acc['dice'])}\n")
        f.write(f"Recall: {np.mean(metrics_acc['recall'])}\n")

if __name__ == "__main__":
    run_evaluation()