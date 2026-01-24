import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from model_drunet import DRUNet

# ================= ⚙️ CONFIGURATION =================
# Updated Path from your input
DATASET_2019_PATH = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training"
MODEL_PATH = "results/drunet_highcap_best.pth.tar"
RESULTS_DIR = "results/generalization_2019"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

def calculate_dice(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def run_2019_audit():
    print(f"🚀 Starting Generalization Audit on BraTS 2019...")
    
    # 1. Load DRUNet Model
    print(f"🔄 Loading High-Capacity DRUNet on {DEVICE}...")
    model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Scanning Folders (HGG & LGG Logic from your old script)
    folders = ['HGG', 'LGG']
    patient_scores = []
    
    # Global Metrics Accumulators
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
    
    # Best/Worst Case Tracking
    best_dice = 0.0; worst_dice = 1.0
    best_sample = None; worst_sample = None

    total_slices_processed = 0

    for sub_folder in folders:
        category_path = os.path.join(DATASET_2019_PATH, sub_folder)
        if not os.path.exists(category_path):
            print(f"⚠️ Warning: Folder {sub_folder} not found, skipping...")
            continue
        
        patients = sorted(os.listdir(category_path))
        print(f"📂 Processing {len(patients)} patients in {sub_folder}...")

        for patient in tqdm(patients):
            patient_path = os.path.join(category_path, patient)
            
            # Smart File Handling (.nii vs .nii.gz)
            flair_file = os.path.join(patient_path, f"{patient}_flair.nii")
            seg_file = os.path.join(patient_path, f"{patient}_seg.nii")
            
            if not os.path.exists(flair_file): flair_file += ".gz"
            if not os.path.exists(seg_file): seg_file += ".gz"

            if not (os.path.exists(flair_file) and os.path.exists(seg_file)):
                continue

            # Load Volumes
            try:
                flair_data = nib.load(flair_file).get_fdata()
                seg_data = nib.load(seg_file).get_fdata()
            except Exception as e:
                continue

            # Process Middle Slices (ROI) to save time
            # Using slices 60-110 where tumor is most likely to avoid empty slice bias
            for slice_idx in range(60, 110): 
                img_slice = flair_data[:, :, slice_idx]
                mask_slice = seg_data[:, :, slice_idx]
                
                # Convert multiclass mask to binary (Tumor vs Background)
                mask_slice = (mask_slice > 0).astype(np.float32)

                # Skip empty slices to focus on Tumor Detection capability
                if np.sum(mask_slice) < 10: continue

                # Preprocessing (Resize & Normalize)
                img_input = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
                if np.max(img_input) > 0: 
                    img_input = (img_input - np.mean(img_input)) / (np.std(img_input) + 1e-8)
                
                # Prepare Tensor
                img_tensor = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Inference
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        pred_logits = model(img_tensor)
                        pred_probs = torch.sigmoid(pred_logits)
                    pred_mask = (pred_probs > 0.5).float().squeeze().cpu().numpy()

                # Resize Ground Truth to 256x256 to match prediction
                true_mask = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

                # Metrics Calculation
                dice = calculate_dice(pred_mask, true_mask)
                patient_scores.append(dice)
                total_slices_processed += 1

                # Confusion Matrix Updates
                tp_total += np.sum((pred_mask == 1) & (true_mask == 1))
                tn_total += np.sum((pred_mask == 0) & (true_mask == 0))
                fp_total += np.sum((pred_mask == 1) & (true_mask == 0))
                fn_total += np.sum((pred_mask == 0) & (true_mask == 1))

                # Track Visuals
                if dice > best_dice:
                    best_dice = dice
                    best_sample = (img_input, true_mask, pred_mask)
                if dice < worst_dice and dice > 0.05:
                    worst_dice = dice
                    worst_sample = (img_input, true_mask, pred_mask)

    if total_slices_processed == 0:
        print("❌ Error: No valid slices found. Check dataset path.")
        return

    # ================= 📊 GENERATE PLOTS =================
    
    # 1. Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=patient_scores, color='lightgreen', showmeans=True)
    plt.title(f"Generalization: BraTS 2019 (N={total_slices_processed} slices)")
    plt.ylabel("Dice Score")
    plt.savefig(f"{RESULTS_DIR}/generalization_boxplot.png")
    
    # 2. Confusion Matrix
    cm = np.array([[int(tn_total), int(fp_total)], [int(fn_total), int(tp_total)]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix: BraTS 2019")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(f"{RESULTS_DIR}/generalization_confusion_matrix.png")

    # 3. Best/Worst Cases
    def save_vis(sample, name, score):
        if sample is None: return
        img, true, pred = sample
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.title("MRI (Normalized)"); plt.imshow(img, cmap='gray')
        plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(true, cmap='jet')
        plt.subplot(1, 3, 3); plt.title(f"Prediction (Dice: {score:.2f})"); plt.imshow(pred, cmap='jet')
        plt.savefig(f"{RESULTS_DIR}/{name}.png")
        plt.close()

    save_vis(best_sample, "best_case_2019", best_dice)
    save_vis(worst_sample, "worst_case_2019", worst_dice)

    # Report
    recall = tp_total / (tp_total + fn_total + 1e-6)
    fpr = fp_total / (tn_total + fp_total + 1e-6)
    mean_dice = np.mean(patient_scores)

    print("\n✅ BraTS 2019 Audit Complete!")
    print(f"🔹 Mean Dice Score: {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"🔹 Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
    print(f"🔹 False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"📂 Plots saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_2019_audit()