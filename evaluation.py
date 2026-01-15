import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import UNET
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================= SETTINGS =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "BraTS_Split/test"
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
BATCH_SIZE = 32  # Evaluation is faster, so we can increase batch size

def load_checkpoint(checkpoint_file, model):
    print("=> Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])

def get_confusion_matrix_elements(preds, targets):
    """
    Calculates TP, TN, FP, FN for binary segmentation on GPU
    """
    # 1 = Tumor, 0 = Background
    TP = (preds * targets).sum().item()
    TN = ((1 - preds) * (1 - targets)).sum().item()
    FP = (preds * (1 - targets)).sum().item()
    FN = ((1 - preds) * targets).sum().item()
    return TP, TN, FP, FN

def evaluate_model():
    print(f"🚀 Starting Full Evaluation on {DEVICE}...")
    
    # 1. Load Model
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    if os.path.exists(CHECKPOINT_FILE):
        load_checkpoint(CHECKPOINT_FILE, model)
    else:
        print("❌ Error: Checkpoint file not found!")
        return

    model.eval()

    # 2. Load Test Data
    test_ds = BrainTumorDataset(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    
    print(f"📂 Evaluating on {len(test_ds)} test images...")

    # Metrics storage
    total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
    total_dice = 0
    
    # 3. Scanning Loop
    loop = tqdm(test_loader)
    with torch.no_grad():
        for x, y in loop:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Predict
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Calculate Confusion Matrix Elements for this batch
            TP, TN, FP, FN = get_confusion_matrix_elements(preds, y)
            
            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN

            # Dice Score for this batch
            smooth = 1e-8
            dice = (2 * TP) / (2 * TP + FP + FN + smooth)
            total_dice += dice

    # 4. Final Calculations
    avg_dice = total_dice / len(test_loader)
    
    # Precision (When model says tumor, how often is it right?)
    precision = total_TP / (total_TP + total_FP + 1e-8)
    
    # Recall / Sensitivity (How many actual tumors did we catch?) -> MOST IMPORTANT
    recall = total_TP / (total_TP + total_FN + 1e-8)
    
    # Accuracy (Overall correctness, usually high because background is large)
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

    print("\n" + "="*30)
    print("🏥 FINAL MEDICAL REPORT")
    print("="*30)
    print(f"✅ Dice Score (Overall Quality): {avg_dice:.4f}")
    print(f"🎯 Pixel Accuracy:              {accuracy*100:.2f}%")
    print(f"🔍 Precision (Reliability):     {precision:.4f}")
    print(f"🚑 Recall (Sensitivity):        {recall:.4f} (Ability to catch tumor)")
    print("-" * 30)
    print("RAW PIXEL COUNTS:")
    print(f"True Positives (Tumor Found):   {int(total_TP)}")
    print(f"False Negatives (Missed):       {int(total_FN)}")
    print(f"False Positives (False Alarm):  {int(total_FP)}")
    print("="*30)

    # 5. Plot Confusion Matrix
    plot_confusion_matrix(total_TP, total_TN, total_FP, total_FN)

def plot_confusion_matrix(TP, TN, FP, FN):
    # Normalize to percentages for better readability
    total = TP + TN + FP + FN
    
    # Data for heatmap
    matrix_data = np.array([[TN, FP], [FN, TP]])
    matrix_percents = matrix_data / total
    
    labels = [
        [f"True Negative\n(Clear)\n{matrix_percents[0,0]:.2%}", f"False Positive\n(False Alarm)\n{matrix_percents[0,1]:.2%}"],
        [f"False Negative\n(Missed Tumor)\n{matrix_percents[1,0]:.2%}", f"True Positive\n(Caught Tumor)\n{matrix_percents[1,1]:.2%}"]
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_data, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Predicted Clear', 'Predicted Tumor'],
                yticklabels=['Actual Clear', 'Actual Tumor'])
    
    plt.title("Confusion Matrix (Pixel Level Evaluation)")
    plt.ylabel('Ground Truth (Doctor)')
    plt.xlabel('AI Prediction')
    
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\n📊 Confusion Matrix saved as '{save_path}'")

import os
if __name__ == "__main__":
    evaluate_model()