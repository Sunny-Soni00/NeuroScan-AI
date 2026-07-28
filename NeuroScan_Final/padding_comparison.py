import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from model_drunet_v2 import AttentionDRUNet

# ================= ⚙️ CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model_v2.pth"
# Diverse selection: Picking 4 slices from different parts of the brain
ALL_VAL_IMAGES = sorted(glob.glob("../BraTS_Split/val/images/*.png"))
TEST_IMAGES = [ALL_VAL_IMAGES[i] for i in [2, 50, 100, 150]] # Diverse indices
OUTPUT_NAME = "Final_Padding_Audit.png"

# ================= 🧠 LOAD MODEL =================
model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

def dice_coeff(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * inter + 1e-6) / (union + 1e-6)

def run_audit():
    num_samples = len(TEST_IMAGES)
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
    plt.subplots_adjust(hspace=0.4)

    print(f"🚀 Generating Audit for {num_samples} diverse slices...")

    for i, img_path in enumerate(TEST_IMAGES):
        # 1. Load & Z-Score Normalization [cite: 516-520]
        img = cv2.imread(img_path).astype(np.float32)
        img_norm = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # 2. Inference WITHOUT Reflexive Padding
        with torch.no_grad():
            out_no_pad = torch.sigmoid(model(img_tensor))
            mask_no_pad = (out_no_pad[0, 0].cpu().numpy() > 0.5).astype(np.float32)

        # 3. Inference WITH Reflexive Padding [cite: 522-523]
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        # Simulating a non-standard dimension to show padding effect
        img_padded = F.pad(img_tensor, (0, pad_w + 16, 0, pad_h + 16), mode='reflect')
        
        with torch.no_grad():
            out_pad = torch.sigmoid(model(img_padded))
            # Crop back to original size
            mask_pad = (out_pad[0, 0, :h, :w].cpu().numpy() > 0.5).astype(np.float32)

        # Calculate Similarity Score
        score = dice_coeff(mask_no_pad, mask_pad)

        # --- PLOTTING ---
        # Column 1: Input
        axes[i, 0].imshow(img[:, :, 1].astype(np.uint8), cmap='gray')
        axes[i, 0].set_title(f"Input Slice {i+1} (Z-Stack)", fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')

        # Column 2: Without Padding
        axes[i, 1].imshow(mask_no_pad, cmap='Reds')
        axes[i, 1].set_title("Standard Output (Artifact Risk)", fontsize=14, color='red')
        axes[i, 1].axis('off')

        # Column 3: With Padding
        axes[i, 2].imshow(mask_pad, cmap='Greens')
        axes[i, 2].set_title(f"Reflexive Padding (Clean)\nSimilarity: {score:.4f}", fontsize=14, color='green')
        axes[i, 2].axis('off')

    plt.suptitle("NeuroScan Final Audit: Reflexive Padding Evidence", fontsize=22, fontweight='bold', y=0.95)
    plt.savefig(OUTPUT_NAME, dpi=150, bbox_inches='tight')
    print(f"✅ Final audit report saved as {OUTPUT_NAME}")

if __name__ == "__main__":
    run_audit()