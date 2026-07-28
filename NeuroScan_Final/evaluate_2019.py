import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Core Architecture Import
from model_drunet_v2 import AttentionDRUNet 

# ================= ⚙️ CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_2019_PATH = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training"
MODEL_PATH = "checkpoints/best_model_v2.pth"
RESULTS_DIR = "Final_Generalization_Report_2019"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= 📊 1. INTERNAL METRICS ENGINE =================
def get_comprehensive_metrics(pred, target, threshold=0.5):
    """Full math for metrics without external utils."""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    
    return {
        "dice": dice, "iou": iou, 
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

# ================= 🌍 2. DATASET (Z-Score + Reflexive Padding) =================
class BraTS2019_FullInference(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for sub in ['HGG', 'LGG']:
            sub_path = os.path.join(root_dir, sub)
            if not os.path.exists(sub_path): continue
            for patient in os.listdir(sub_path):
                p_path = os.path.join(sub_path, patient)
                flair = [f for f in os.listdir(p_path) if 'flair' in f][0]
                seg = [f for f in os.listdir(p_path) if 'seg' in f][0]
                # ROI focus: Middle slices
                for z in range(60, 100, 2): 
                    self.samples.append({'f': os.path.join(p_path, flair), 's': os.path.join(p_path, seg), 'z': z})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        flair_vol = nib.load(s['f']).get_fdata()
        seg_vol = nib.load(s['s']).get_fdata()
        
        # 2.5D Stack logic
        stack = []
        for offset in [-1, 0, 1]:
            img = cv2.resize(flair_vol[:, :, s['z'] + offset], (256, 256))
            img = (img - np.mean(img)) / (np.std(img) + 1e-8) # Z-Score
            stack.append(img)
            
        img_tensor = torch.from_numpy(np.stack(stack, axis=0)).float()
        mask = cv2.resize(seg_vol[:, :, s['z']], (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        
        # Reflexive Padding for UNet
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        return img_tensor, mask_tensor

# ================= 📈 3. PLOTTING & % MATRIX ENGINE =================
def save_normalized_cm(tp, fp, fn, tn):
    """Saves Confusion Matrix with both Numbers and Percentages."""
    cm = np.array([[tn, fp], [fn, tp]])
    cm_perc = cm.astype('float') / cm.sum() * 100 # Total % normalization
    
    labels = [f"{v}\n({p:.2f}%)" for v, p in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.array(labels).reshape(2, 2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=True,
                xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
    plt.xlabel('NeuroScan Prediction'); plt.ylabel('Actual Label')
    plt.title('BraTS 2019 Global Confusion Matrix (Pixels & %)')
    plt.savefig(f"{RESULTS_DIR}/normalized_confusion_matrix.png")
    plt.close()

def save_visual_report(img, gt, pred, name):
    """Side-by-side comparison for project showcase."""
    img_display = img[1].cpu().numpy() # Middle slice of 2.5D stack
    gt_display = gt[0].cpu().numpy()
    pred_display = (torch.sigmoid(pred[0]) > 0.5).float().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img_display, cmap='gray'); plt.title("Input MRI (FLAIR)")
    plt.subplot(1, 3, 2); plt.imshow(gt_display, cmap='gray'); plt.title("Ground Truth (Expert)")
    plt.subplot(1, 3, 3); plt.imshow(pred_display, cmap='gray'); plt.title("NeuroScan Prediction")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{name}.png")
    plt.close()

# ================= 🚀 4. MAIN EXECUTION =================
def run_full_evaluation():
    # 1. Load Model
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # 2. Prepare Data
    ds = BraTS2019_FullInference(DATASET_2019_PATH)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    
    all_results = []
    total_px = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    print(f"🧪 Testing Generalization on BraTS 2019 Raw Data...")
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                preds = model(x)
            
            # Per-batch metrics for CSV
            m = get_comprehensive_metrics(preds, y)
            all_results.append(m)
            
            # Global pixel accumulation for Confusion Matrix
            for k in total_px: total_px[k] += m[k]

            # Qualitative visual check every 50 batches
            if i % 50 == 0:
                save_visual_report(x[0], y[0], preds[0], f"qualitative_slice_batch_{i}")

    # 3. Final Aggregation & Plotting
    df = pd.DataFrame(all_results)
    final_avg = df.mean().to_dict()
    
    print(f"\n🏆 Results for BraTS 2019:")
    print(f"Dice: {final_avg['dice']:.4f} | Recall: {final_avg['recall']:.4f}")
    
    save_normalized_cm(total_px['tp'], total_px['fp'], total_px['fn'], total_px['tn'])
    
    # Save Metrics Plot
    metrics_to_plot = {k: final_avg[k] for k in ['dice', 'iou', 'precision', 'recall']}
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color='darkred')
    plt.ylim(0, 1); plt.title("Final Generalization Scores"); plt.ylabel("Accuracy Score")
    plt.savefig(f"{RESULTS_DIR}/summary_metrics_chart.png")

    print(f"✅ Full report generated in {RESULTS_DIR}/")

if __name__ == "__main__":
    run_full_evaluation()