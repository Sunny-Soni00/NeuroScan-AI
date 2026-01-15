import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import UNET
import cv2

# ================= ⚙️ CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
DATASET_2019_PATH = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training" 
IMG_SIZE = 256

# Metrics helper functions
def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def get_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn

# ================= 🧠 MODEL LOADING (FIXED PART) =================
print(f"🔄 Loading model on {DEVICE}...")
model = UNET(in_channels=1, out_channels=1).to(DEVICE)

# Checkpoint load karna
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("✅ Model weights loaded successfully!")
else:
    print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
    exit()

# ================= 📂 DATA SCANNING =================
all_dice_scores = []
cm_total = np.zeros(4) # [TP, FP, FN, TN]
folders = ['HGG', 'LGG']

for sub_folder in folders:
    category_path = os.path.join(DATASET_2019_PATH, sub_folder)
    if not os.path.exists(category_path):
        continue
    
    patients = os.listdir(category_path)
    for patient in tqdm(patients, desc=f"Auditing {sub_folder}"):
        patient_path = os.path.join(category_path, patient)
        
        # File path logic
        flair_file = os.path.join(patient_path, f"{patient}_flair.nii")
        seg_file = os.path.join(patient_path, f"{patient}_seg.nii")
        
        if not os.path.exists(flair_file):
            flair_file += ".gz"
            seg_file += ".gz"

        if not (os.path.exists(flair_file) and os.path.exists(seg_file)):
            continue

        # Process 3D Volume
        flair_data = nib.load(flair_file).get_fdata()
        seg_data = nib.load(seg_file).get_fdata()

        for slice_idx in range(50, 120):
            img_slice = flair_data[:, :, slice_idx]
            mask_slice = (seg_data[:, :, slice_idx] > 0).astype(np.float32)
            
            if np.sum(mask_slice) < 50: continue

            # Preprocess & Predict
            img_input = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
            # Normalization
            if np.max(img_input) > 0: img_input /= np.max(img_input)
            
            img_tensor = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = torch.sigmoid(model(img_tensor))
                pred = (pred > 0.5).float().squeeze().cpu().numpy()

            # Final Evaluation
            true_mask = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            all_dice_scores.append(dice_coef(true_mask, pred))
            tp, fp, fn, tn = get_metrics(true_mask, pred)
            cm_total += [tp, fp, fn, tn]

# ================= 📊 SAVE FINAL ANALYSIS =================
# Histogram
plt.figure(figsize=(10, 6))
plt.hist(all_dice_scores, bins=25, color='green', alpha=0.6)
plt.axvline(np.mean(all_dice_scores), color='red', linestyle='--')
plt.title(f"Generalization Check: Mean Dice {np.mean(all_dice_scores):.4f}")
plt.savefig("generalization_histogram.png")

print(f"\n🏁 Finished! Mean Dice on 2019 Dataset: {np.mean(all_dice_scores):.4f}")