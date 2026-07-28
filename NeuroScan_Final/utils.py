import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# Global configuration for plots
plt.rcParams.update({'font.size': 10})

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculates Precision, Recall, and IoU with safety detachments.
    """
    # Detach and thresholding
    pred = (torch.sigmoid(pred).detach() > threshold).float()
    target = target.detach().float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    
    return precision.item(), recall.item(), iou.item()

def save_training_plots(log_path, output_dir):
    """
    Generates training curves with column validation and directory safety.
    """
    if not os.path.exists(log_path):
        print(f"⚠️ Warning: Log file {log_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(log_path)
    
    # Check if required columns exist to avoid KeyError
    required_cols = ['train_loss', 'val_loss', 'dice']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Warning: CSV missing one of {required_cols}. Skipping plots.")
        return

    plt.figure(figsize=(12, 5))
    
    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(df['train_loss'], label='Train')
    plt.plot(df['val_loss'], label='Val')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Dice Subplot
    plt.subplot(1, 2, 2)
    plt.plot(df['dice'], label='Val Dice', color='green')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary.png"), dpi=300)
    plt.close()

def save_confusion_matrix(pred, target, output_path, labels=['Healthy', 'Tumor']):
    """
    Generates pixel-level confusion matrix with proper detachment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to flattened numpy arrays safely
    pred_flat = (torch.sigmoid(pred).detach() > 0.5).cpu().numpy().astype(np.uint8).flatten()
    target_flat = target.detach().cpu().numpy().astype(np.uint8).flatten()
    
    cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Pixel-Level Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def save_visual_comparison(img, mask, pred, output_path):
    """
    Visualizes MRI vs Ground Truth vs Prediction with shape validation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Shape validation: Expecting [B, C, H, W]
    if img.dim() != 4 or mask.dim() != 4:
        print("⚠️ Warning: Visualizer expects 4D tensors [B, C, H, W]. Skipping.")
        return

    # Safe conversion to numpy
    img_np = img.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    pred_np = (torch.sigmoid(pred).detach() > 0.5).cpu().numpy()

    # Selecting the middle slice of the batch for visualization
    idx = 0 
    # Use middle channel if 3-channel input, else first channel
    c_idx = img_np.shape[1] // 2 

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np[idx, c_idx, :, :], cmap='gray')
    axes[0].set_title('Input MRI')
    
    axes[1].imshow(mask_np[idx, 0, :, :], cmap='gray')
    axes[1].set_title('Ground Truth')
    
    axes[2].imshow(pred_np[idx, 0, :, :], cmap='gray')
    axes[2].set_title('NeuroScan Prediction')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()