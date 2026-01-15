import torch
import torch.nn as nn
import cv2
import os
import matplotlib
matplotlib.use('Agg') # Screen error rokne ke liye
import matplotlib.pyplot as plt
from model import UNET
from dataset import BrainTumorDataset
from torch.utils.data import DataLoader

# ================= SETTINGS =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "BraTS_Split/test" # Final Exam Data
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
OUTPUT_FOLDER = "final_results"

def load_checkpoint(checkpoint_file, model):
    print("=> Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])

def save_comparison_images(loader, model, folder="final_results", device="cuda"):
    model.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Generating Results... (Might take 1-2 mins)")
    
    # Sirf pehla batch (16 images) lenge result dikhane ke liye
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # Binary Mask (0 or 1)
        
        # CPU pe laao plot karne ke liye
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        preds = preds.cpu().numpy()

        # 5 Best Examples Plot karenge
        plt.figure(figsize=(15, 10))
        for i in range(5):
            # 1. Original MRI
            plt.subplot(5, 3, i*3 + 1)
            plt.imshow(x[i][0], cmap='gray')
            plt.title("Original MRI")
            plt.axis("off")

            # 2. Asli Mask (Doctor wala)
            plt.subplot(5, 3, i*3 + 2)
            plt.imshow(y[i][0], cmap='gray')
            plt.title("Actual Tumor (Doctor)")
            plt.axis("off")

            # 3. AI Prediction (Tumhara Model)
            plt.subplot(5, 3, i*3 + 3)
            plt.imshow(preds[i][0], cmap='gray')
            plt.title("AI Predicted Tumor")
            plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/Final_Report_Card.png")
        print(f"✅ Result Saved: Check '{folder}/Final_Report_Card.png'")
        break # Sirf ek batch kafi hai

def main():
    print(f"🚀 Loading Best Model (90% Accuracy)...")
    
    # Model Waisa hi initialize karo jaisa training mein tha
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    
    # Load Saved Weights
    if os.path.exists(CHECKPOINT_FILE):
        load_checkpoint(CHECKPOINT_FILE, model)
    else:
        print("❌ Checkpoint file nahi mili! Kya pichla code save hua tha?")
        return

    # Test Data Load karo
    test_ds = BrainTumorDataset(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=True)

    # Report Generate karo
    save_comparison_images(test_loader, model, OUTPUT_FOLDER, DEVICE)

if __name__ == "__main__":
    main()