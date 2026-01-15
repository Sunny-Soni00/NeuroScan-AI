import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import UNET
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "BraTS_Split/test"
CHECKPOINT_FILE = "my_checkpoint.pth.tar"

def save_worst_cases():
    print("🔍 Hunting for Mistakes (Worst Predictions)...")
    
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_ds = BrainTumorDataset(TEST_IMG_DIR)
    # Batch size 1 zaroori hai individual check ke liye
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) 

    worst_scores = [] # Store (score, image, mask, pred)

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Agar tumor hi nahi hai, toh skip karo (hum sirf tumor misses dhund rahe hain)
            if torch.sum(y) == 0:
                continue

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Calculate Score
            intersection = (preds * y).sum()
            score = (2. * intersection) / (preds.sum() + y.sum() + 1e-8)
            score = score.item()

            # Agar score kam hai, list mein daalo
            if score < 0.5: 
                # CPU pe convert karke save kar lo
                img_np = x[0][0].cpu().numpy()
                mask_np = y[0][0].cpu().numpy()
                pred_np = preds[0][0].cpu().numpy()
                worst_scores.append((score, img_np, mask_np, pred_np))

    # Sort by score (Lowest first) and take top 5
    worst_scores.sort(key=lambda x: x[0])
    top_5_worst = worst_scores[:5]

    print(f"Found {len(worst_scores)} bad predictions out of thousands. Showing top 5 worst.")

    # Plotting
    if len(top_5_worst) > 0:
        plt.figure(figsize=(15, 10))
        for i, (score, img, mask, pred) in enumerate(top_5_worst):
            plt.subplot(5, 3, i*3 + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"MRI (Worst #{i+1})")
            plt.axis('off')

            plt.subplot(5, 3, i*3 + 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Doctor's Mask")
            plt.axis('off')

            plt.subplot(5, 3, i*3 + 3)
            plt.imshow(pred, cmap='gray')
            plt.title(f"AI Fail (Score: {score:.2f})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("analysis_worst_cases.png")
        print("✅ Saved 'analysis_worst_cases.png'. Dekho kahan galti hui.")
    else:
        print("🎉 Great News! No predictions below 0.5 found. Model is super robust!")

if __name__ == "__main__":
    save_worst_cases()