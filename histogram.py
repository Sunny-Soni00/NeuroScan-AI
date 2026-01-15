import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import UNET
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= SETTINGS =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "BraTS_Split/test"
CHECKPOINT_FILE = "my_checkpoint.pth.tar"

def get_dice_score(preds, targets):
    smooth = 1e-8
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + smooth)

def analyze_stability():
    print("📊 Generating Stability Histogram...")
    
    # Load Model
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_ds = BrainTumorDataset(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch size 1 taaki har image ka alag score mile

    scores = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Sirf un images ka score lo jisme sach mein Tumor tha (Khali images ignore karo)
            if torch.sum(y) > 0:
                score = get_dice_score(preds, y).item()
                scores.append(score)

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, color='green')
    plt.title(f'Model Stability Check (Total Tumor Images: {len(scores)})')
    plt.xlabel('Dice Score (Accuracy)')
    plt.ylabel('Number of Patients')
    plt.axvline(x=0.85, color='r', linestyle='--', label='Excellent Mark (0.85)')
    plt.legend()
    
    plt.savefig("analysis_histogram.png")
    print("✅ Saved 'analysis_histogram.png'. Check karo graph kaisa hai.")

if __name__ == "__main__":
    analyze_stability()