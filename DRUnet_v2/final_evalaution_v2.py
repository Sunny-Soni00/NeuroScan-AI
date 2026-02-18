import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your model & utils
from dataset_balance_v2 import BraTSDataset25D
from model_drunet_v2 import AttentionDRUNet
from utils_v2 import calculate_all_metrics

# ================= ⚙️ CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Path adjusted to where your Split folder is
INTERNAL_TEST_PATH = "../../BraTS_Split/test" 
# Raw NIfTI Data Path for 2019
DATASET_2019_PATH = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training"
# Using the model checkpoint you requested (Make sure this file exists!)
MODEL_PATH = "results/drunet_highcap_best.pth.tar" 
# Fallback to V2 checkpoint if above not found
MODEL_PATH_V2 = "results/v2_checkpoint.pth.tar"

RESULTS_DIR = "results/generalization_2019"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= 📊 1. PLOTTING FUNCTION =================
def plot_final_training_graphs():
    log_path = "results/v2_metrics_log.csv"
    if not os.path.exists(log_path):
        print("⚠️ No CSV log found. Skipping plots.")
        return

    df = pd.read_csv(log_path)
    epochs = range(1, len(df) + 1)

    plt.figure(figsize=(18, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, df['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, df['val_loss'], 'r-s', label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # Dice & IoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, df['dice'], 'g-^', label='Dice Score')
    plt.plot(epochs, df['iou'], 'm-x', label='IoU Score')
    plt.title('Segmentation Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Score'); plt.legend(); plt.grid(True)

    # Precision & Recall
    plt.subplot(1, 3, 3)
    plt.plot(epochs, df['precision'], 'c-d', label='Precision')
    plt.plot(epochs, df['recall'], 'y-v', label='Recall')
    plt.title('Precision vs Recall'); plt.xlabel('Epochs'); plt.ylabel('Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/final_training_plots.png")
    print(f"✅ Training plots saved to {RESULTS_DIR}/final_training_plots.png")

# ================= 🧪 2. INTERNAL TEST (BraTS 2021) =================
def evaluate_internal_test(model):
    print(f"\n🚀 Evaluating on Internal Test Set (BraTS 2021 held-out 10%)...")
    test_ds = BraTSDataset25D(INTERNAL_TEST_PATH)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            with torch.amp.autocast('cuda'):
                preds = model(x)
            
            # Using TTA (Test Time Augmentation) for best results
            # preds = (model(x) + torch.flip(model(torch.flip(x, [3])), [3])) / 2.0 
            
            m = calculate_all_metrics(preds, y)
            for k in metrics: metrics[k].append(m[k])

    print(f"🏆 Internal Test Results:")
    print(f"   Dice: {np.mean(metrics['dice']):.4f}")
    print(f"   IoU : {np.mean(metrics['iou']):.4f}")

# ================= 🌍 3. EXTERNAL GENERALIZATION (BraTS 2019) =================
class BraTS2019_DirectLoader(Dataset):
    """
    Reads Raw NIfTI files from BraTS 2019, extracts relevant slices on-the-fly,
    and creates 2.5D stacks. No pre-processing needed!
    """
    def __init__(self, root_dir):
        self.samples = []
        # Walk through HGG and LGG folders
        for sub in ['HGG', 'LGG']:
            sub_path = os.path.join(root_dir, sub)
            if not os.path.exists(sub_path): continue
            
            for patient in os.listdir(sub_path):
                p_path = os.path.join(sub_path, patient)
                # Find FLAIR and Seg files
                flair_file = [f for f in os.listdir(p_path) if 'flair' in f][0]
                seg_file = [f for f in os.listdir(p_path) if 'seg' in f][0]
                
                # We store path and slices of interest (Middle 60-100 where tumor is likely)
                # Loading ALL 155 slices is too slow for quick eval, focusing on ROI
                for z in range(60, 100, 2): 
                    self.samples.append({
                        'flair': os.path.join(p_path, flair_file),
                        'seg': os.path.join(p_path, seg_file),
                        'z': z
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z = sample['z']
        
        # Load Volume (Slow but works for raw eval)
        flair_vol = nib.load(sample['flair']).get_fdata()
        seg_vol = nib.load(sample['seg']).get_fdata()
        
        # Create 2.5D Stack (Z-1, Z, Z+1)
        stack = []
        for offset in [-1, 0, 1]:
            slice_img = flair_vol[:, :, z + offset]
            # Resize to 256x256 (Model Input)
            slice_img = cv2.resize(slice_img, (256, 256))
            # Normalize
            if np.max(slice_img) > 0:
                slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
            stack.append(slice_img)
            
        img_stack = np.stack(stack, axis=0).astype(np.float32) # (3, 256, 256)
        
        # Ground Truth
        mask = seg_vol[:, :, z]
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32) # Binarize (Tumor vs Background)
        
        return torch.tensor(img_stack), torch.tensor(mask)

def evaluate_external_2019(model):
    print(f"\n🌍 Evaluating on External Generalization Set (BraTS 2019 NIfTI)...")
    if not os.path.exists(DATASET_2019_PATH):
        print(f"❌ Path not found: {DATASET_2019_PATH}")
        return

    ds_2019 = BraTS2019_DirectLoader(DATASET_2019_PATH)
    if len(ds_2019) == 0:
        print("❌ No samples found. Check folder structure (HGG/LGG).")
        return
        
    loader = DataLoader(ds_2019, batch_size=8, shuffle=False, num_workers=2) # Batch 8 for NIfTI loading safety
    
    metrics = {"dice": [], "recall": []}
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Processing 2019 Data"):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            with torch.amp.autocast('cuda'):
                preds = model(x)
            
            m = calculate_all_metrics(preds, y)
            metrics['dice'].append(m['dice'])
            metrics['recall'].append(m['recall'])
            
    print(f"🏆 BraTS 2019 Generalization Results:")
    print(f"   Dice Score: {np.mean(metrics['dice']):.4f}")
    print(f"   Recall    : {np.mean(metrics['recall']):.4f}")
    
    # Save simple report
    with open(f"{RESULTS_DIR}/2019_report.txt", "w") as f:
        f.write(f"BraTS 2019 Evaluation\nDice: {np.mean(metrics['dice'])}\nRecall: {np.mean(metrics['recall'])}")

# ================= 🚀 MAIN EXECUTION =================
if __name__ == "__main__":
    # 1. Load Model
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    final_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_V2
    
    if os.path.exists(final_model_path):
        print(f"✅ Loading Model from: {final_model_path}")
        checkpoint = torch.load(final_model_path)
        # Handle state_dict key mismatch if any
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        print(f"❌ Model Checkpoint not found at {MODEL_PATH} or {MODEL_PATH_V2}")
        exit()

    # 2. Run Steps
    plot_final_training_graphs()
    evaluate_internal_test(model)
    evaluate_external_2019(model)