import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Create the results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_checkpoint(state, filename="drunet_checkpoint.pth.tar"):
    """Saves the current model weights to the results folder."""
    filepath = os.path.join(RESULTS_DIR, filename)
    print(f"=> Saving checkpoint to {filepath}")
    torch.save(state, filepath)

def load_checkpoint(checkpoint, model):
    """Loads weights from a saved checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    """
    Calculates the Dice Score and Pixel Accuracy during training.
    Dice = (2 * intersection) / (total_pixels)
    """
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Dice Score Calculation
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    model.train()
    return dice_score / len(loader)

def plot_learning_curves(train_losses, val_dice_scores):
    """Generates and saves Training vs Validation plots."""
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss (Hybrid)')
    plt.title('Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Dice Score Curve
    plt.subplot(1, 2, 2)
    plt.plot(val_dice_scores, label='Validation Dice', color='green')
    plt.axhline(y=0.92, color='r', linestyle='--', label='Previous Best (0.92)')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'))
    plt.close()