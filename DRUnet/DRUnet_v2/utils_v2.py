import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def save_checkpoint(state, filename="results/v2_checkpoint.pth.tar"):
    torch.save(state, filename)

def calculate_all_metrics(preds, targets, threshold=0.5):
    # Preds are raw logits, applying sigmoid for metric calculation
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > threshold).float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}

def plot_comprehensive_metrics(history, save_path="results"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss"); plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["dice"], label="Dice")
    plt.plot(epochs, history["iou"], label="IoU")
    plt.title("Segmentation Accuracy"); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["precision"], label="Precision")
    plt.plot(epochs, history["recall"], label="Recall")
    plt.title("Metrics"); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "comprehensive_metrics.png"))
    plt.close()

class HybridFocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1e-6):
        # Dice uses Sigmoid
        probs = torch.sigmoid(inputs).view(-1)
        flat_targets = targets.view(-1)
        
        intersection = (probs * flat_targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + flat_targets.sum() + smooth)
        
        # Focal Loss uses raw Logits for Autocast stability
        # binary_cross_entropy_with_logits is the stable version of BCE
        bce_logits = nn.functional.binary_cross_entropy_with_logits(inputs.view(-1), flat_targets, reduction='none')
        pt = torch.exp(-bce_logits)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * bce_logits).mean()
        
        return dice_loss + focal_loss