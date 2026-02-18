import torch
import torch.nn as nn
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Screen error rokne ke liye
import matplotlib.pyplot as plt
from DRUnet.DRUnet_v2.model_drunet_v2 import AttentionDRUNet
from DRUnet.DRUnet_v2.dataset_balance_v2 import BraTSDataset25D
from torch.utils.data import DataLoader

# ================= SETTINGS =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "BraTS_Split/test"  # Final Exam Data
CHECKPOINT_FILE = "DRUnet/DRUnet_v2/results/v2_checkpoint.pth.tar"
OUTPUT_FOLDER = "drunetv2_results"

def load_checkpoint(checkpoint_file, model):
    print("=> Loading DRUnetv2 Checkpoint...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)

def save_comparison_images(loader, model, folder="drunetv2_results", device="cuda"):
    """
    Generates comparison visualizations for DRUnetv2 predictions.
    Shows Original MRI, Actual Tumor Mask (Doctor), and AI Predicted Tumor.
    """
    model.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Generating DRUnetv2 Results... (Might take 1-2 mins)")
    
    # Sirf pehla batch (16 images) lenge result dikhane ke liye
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Binary Mask (0 or 1)
        
        # CPU pe laao plot karne ke liye
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        preds = preds.cpu().numpy()

        # 5 Best Examples Plot karenge
        plt.figure(figsize=(15, 10))
        for i in range(min(5, x.shape[0])):
            # 1. Original MRI (Take middle channel of 2.5D stack)
            plt.subplot(5, 3, i*3 + 1)
            plt.imshow(x[i][1], cmap='gray')  # Middle slice of 2.5D stack
            plt.title("Original MRI")
            plt.axis("off")

            # 2. Asli Mask (Doctor wala)
            plt.subplot(5, 3, i*3 + 2)
            plt.imshow(y[i], cmap='gray')  # y shape is (16, 256, 256) - no channel dim
            plt.title("Actual Tumor (Doctor)")
            plt.axis("off")

            # 3. AI Prediction (DRUnetv2 Model)
            plt.subplot(5, 3, i*3 + 3)
            plt.imshow(preds[i][0], cmap='gray')  # preds shape is (16, 1, 256, 256)
            plt.title("AI Predicted Tumor (DRUnetv2)")
            plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/DRUnetv2_Final_Report_Card.png")
        print(f"✅ Result Saved: Check '{folder}/DRUnetv2_Final_Report_Card.png'")
        break  # Sirf ek batch kafi hai

def main():
    print(f"🚀 Loading DRUnetv2 Model with Best Weights...")
    
    # AttentionDRUNet with 3 input channels (2.5D stack) and 1 output channel (binary mask)
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Load Saved Weights
    if os.path.exists(CHECKPOINT_FILE):
        load_checkpoint(CHECKPOINT_FILE, model)
    else:
        print("❌ Checkpoint file nahi mili! Kya training complete hua tha?")
        print(f"   Expected at: {CHECKPOINT_FILE}")
        return

    # Test Data Load karo (2.5D Dataset)
    test_ds = BraTSDataset25D(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=True)

    # Report Generate karo
    save_comparison_images(test_loader, model, OUTPUT_FOLDER, DEVICE)

if __name__ == "__main__":
    main()
