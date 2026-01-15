import os
import torch
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import UNET

# ================= ⚙️ CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
OUTPUT_DIR = "results"
IMG_SIZE = 256

# Paths for both datasets
PATH_2021 = "/home/sunny/BrainTumor_AI/BraTS2021_Raw" # Update if different
PATH_2019 = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 📊 METRIC CALCULATOR =================
def calculate_metrics(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    return dice, tp, fp, fn, tn

# ================= 🧠 MODEL LOADER =================
def load_model():
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# ================= 📂 EVALUATION ENGINE =================
def run_evaluation(model, dataset_path, name="Dataset"):
    print(f"🔬 Evaluating {name}...")
    dice_scores = []
    total_cm = np.zeros(4) # TP, FP, FN, TN
    
    # Simple logic to find NIfTI files in nested folders
    files_found = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith("flair.nii") or f.endswith("flair.nii.gz"):
                files_found.append(os.path.join(root, f))

    # Testing a sample of 20 patients for speed, change to len(files_found) for full
    for flair_path in tqdm(files_found[:20], desc=f"Testing {name}"):
        seg_path = flair_path.replace("flair", "seg")
        
        flair_data = nib.load(flair_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        # Middle slice evaluation
        slice_idx = flair_data.shape[2] // 2
        img_slice = flair_data[:, :, slice_idx]
        mask_slice = (seg_data[:, :, slice_idx] > 0).astype(np.float32)

        if np.sum(mask_slice) < 50: continue # Skip empty slices

        img_input = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
        if np.max(img_input) > 0: img_input /= np.max(img_input)
        img_tensor = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor))
            pred = (pred > 0.5).float().squeeze().cpu().numpy()

        true_mask = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        d, tp, fp, fn, tn = calculate_metrics(true_mask, pred)
        
        dice_scores.append(d)
        total_cm += [tp, fp, fn, tn]

    return np.mean(dice_scores), total_cm

# ================= 📈 MAIN EXECUTION =================
if __name__ == "__main__":
    my_model = load_model()
    
    # 1. Get REAL results from both datasets
    dice_21, cm_21 = run_evaluation(my_model, PATH_2021, "BraTS 2021 (Internal)")
    dice_19, cm_19 = run_evaluation(my_model, PATH_2019, "BraTS 2019 (External)")

    # 2. SAVE COMPARISON PLOT
    plt.figure(figsize=(10, 6))
    plt.bar(['BraTS 21 (Internal)', 'BraTS 19 (Generalization)'], [dice_21, dice_19], color=['#2ecc71', '#3498db'])
    plt.ylabel('Mean Dice Score')
    plt.title('Real-world Model Generalization Audit')
    plt.savefig(f"{OUTPUT_DIR}/comparison_dice.png")

    # 3. SAVE CONFUSION MATRICES
    for name, cm in [("2021", cm_21), ("2019", cm_19)]:
        tp, fp, fn, tn = cm
        matrix = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name}.png")

    # 4. SAVE TEXT SUMMARY
    with open(f"{OUTPUT_DIR}/summary.txt", "w") as f:
        f.write(f"BRAIN TUMOR AI EVALUATION REPORT\n")
        f.write(f"--------------------------------\n")
        f.write(f"BraTS 2021 Mean Dice: {dice_21:.4f}\n")
        f.write(f"BraTS 2019 Mean Dice: {dice_19:.4f}\n")
        f.write(f"Generalization Drop: {((dice_21-dice_19)/dice_21)*100:.2f}%\n")

    print(f"\n✅ All REAL results saved in the '{OUTPUT_DIR}/' folder!")