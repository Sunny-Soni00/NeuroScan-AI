import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Modular Architecture Import
from model_drunet_v2 import AttentionDRUNet 

# ================= ⚙️ CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_BASE_PATH = "../BraTS_Split/val" # Change this if needed
MODEL_PATH = "checkpoints/best_model_v2.pth"
RESULTS_DIR = "Final_Audit_TTA_Report"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= 🌍 1. DATASET (NeuroScan Standards) =================
class NeuroScanTTA_Dataset(Dataset):
    def __init__(self, root_dir):
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, "images/*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks/*.png")))

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        img = cv2.imread(self.img_paths[i]).astype(np.float32)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Z-Score Normalization (NeuroScan Final Requirement)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # Reflexive Padding for UNet Alignment
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        return img_tensor, mask_tensor

# ================= 🪄 2. THE TTA ENGINE =================
def apply_tta(model, x):
    """
    Standard + H-Flip + V-Flip. 
    Averages sigmoid probabilities for smoother boundaries.
    """
    # 1. Standard Prediction
    out_orig = torch.sigmoid(model(x))

    # 2. Horizontal Flip
    x_h = torch.flip(x, [3])
    out_h = torch.sigmoid(model(x_h))
    out_h = torch.flip(out_h, [3])

    # 3. Vertical Flip
    x_v = torch.flip(x, [2])
    out_v = torch.sigmoid(model(x_v))
    out_v = torch.flip(out_v, [2])

    return (out_orig + out_h + out_v) / 3.0

# ================= 📈 3. METRICS & VISUALS =================
def calculate_metrics_tta(pred_prob, target, threshold=0.5):
    pred = (pred_prob > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    return dice, iou, tp, fp, fn, tn

def save_tta_matrix(tp, fp, fn, tn):
    cm = np.array([[tn, fp], [fn, tp]])
    cm_perc = cm.astype('float') / cm.sum() * 100
    labels = [f"{v}\n({p:.2f}%)" for v, p in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.array(labels).reshape(2, 2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=labels, fmt="", cmap='Greens', 
                xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
    plt.title('NeuroScan FINAL Audit: Confusion Matrix (w/ TTA)')
    plt.savefig(f"{RESULTS_DIR}/tta_confusion_matrix.png")
    plt.close()

# ================= 🚀 4. MAIN AUDIT EXECUTION =================
def run_final_tta_audit():
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    ds = NeuroScanTTA_Dataset(DATA_BASE_PATH)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    
    results = {"dice": [], "iou": []}
    px = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    print(f"🚀 Starting Final TTA Audit on {DEVICE}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Processing TTA")):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                tta_probs = apply_tta(model, x)
            
            dice, iou, tp, fp, fn, tn = calculate_metrics_tta(tta_probs, y)
            results["dice"].append(dice)
            results["iou"].append(iou)
            px["tp"] += tp; px["fp"] += fp; px["fn"] += fn; px["tn"] += tn

    # Summary Plots
    save_tta_matrix(px['tp'], px['fp'], px['fn'], px['tn'])
    
    # Final Report
    final_dice = np.mean(results["dice"])
    final_iou = np.mean(results["iou"])
    
    print("\n" + "="*40)
    print(f"🏆 FINAL NEUROSCAN PERFORMANCE (w/ TTA)")
    print("="*40)
    print(f"🔹 Dice Score: {final_dice*100:.2f}%")
    print(f"🔹 IoU Score : {final_iou*100:.2f}%")
    print("="*40)

    with open(f"{RESULTS_DIR}/tta_report.txt", "w") as f:
        f.write(f"TTA Dice: {final_dice}\nTTA IoU: {final_iou}")
    
    print(f"✅ Audit complete. Check {RESULTS_DIR}/ for plots.")

if __name__ == "__main__":
    run_visual_inference = run_final_tta_audit()