import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model_drunet import DRUNet
from dataset_balance import BraTSDataset
from torch.utils.data import DataLoader

# ================= ⚙️ TEST CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/drunet_highcap_best.pth.tar"
DATA_PATH = "/home/sunny/BrainTumor_AI/BraTS2021_Raw"
OUTPUT_DIR = "results/test_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_test_results():
    # 1. Load Model with High-Capacity weights
    model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Load Validation Data
    all_patients = sorted(os.listdir(DATA_PATH))
    val_patients = all_patients[40:50] # Using the same 10% split
    val_ds = BraTSDataset(root_dir=DATA_PATH, patient_list=val_patients)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    print(f"🔬 Testing DRUNet on {len(val_patients)} validation patients...")

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 10: break # Save only first 10 test slices for review
            
            x = x.to(DEVICE)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Move to CPU for plotting
            img = x[0][0].cpu().numpy()
            mask = y[0].cpu().numpy()
            pred = preds[0][0].cpu().numpy()

            # Plot Side-by-Side (Original, Ground Truth, Model Prediction)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original MRI (FLAIR)")
            plt.imshow(img, cmap="gray")
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth (Target)")
            plt.imshow(mask, cmap="jet")
            
            plt.subplot(1, 3, 3)
            plt.title("DRUNet Prediction")
            plt.imshow(pred, cmap="jet")

            plt.savefig(os.path.join(OUTPUT_DIR, f"test_result_{i}.png"))
            plt.close()

    print(f"✅ Testing complete. Results saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    visualize_test_results()