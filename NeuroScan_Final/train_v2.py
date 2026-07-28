import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import pandas as pd
import cv2 
from tqdm import tqdm

# Modular Imports
from model_drunet_v2 import AttentionDRUNet 
from losses import HybridFocalDiceLoss 
from utils import (
    calculate_metrics, 
    save_training_plots, 
    save_visual_comparison, 
    save_confusion_matrix
)

# --- CONFIGURATION ---
CONFIG = {
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "batch_size": 4,           
    "total_epochs": 30,
    "patience_scheduler": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "checkpoint_path": "checkpoints/best_model_v2.pth",
    "log_path": "logs/training_metrics.csv",
    "plots_dir": "plots/"
}

# --- DATASET WITH REFLEXIVE PADDING & Z-SCORE ---
class NeuroScanDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        img = cv2.imread(self.img_paths[i])
        if img is None: raise FileNotFoundError(f"Missing image: {self.img_paths[i]}")
        img = img.astype(np.float32)
        
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Missing mask: {self.mask_paths[i]}")
        mask = mask.astype(np.float32) / 255.0
        
        # FEATURE 1: Z-Score Normalization
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # FEATURE 2: Reflexive Padding for UNet (Multiples of 16)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        return img_tensor, mask_tensor

# --- MAIN TRAINING ENGINE ---
def run_full_training():
    for d in ["checkpoints", "logs", "plots"]: os.makedirs(d, exist_ok=True)

    # 1. Setup Data
    base_path = "../BraTS_Split"
    train_imgs = sorted(glob.glob(os.path.join(base_path, "train/images/*.png")))
    train_masks = sorted(glob.glob(os.path.join(base_path, "train/masks/*.png")))
    val_imgs = sorted(glob.glob(os.path.join(base_path, "val/images/*.png")))
    val_masks = sorted(glob.glob(os.path.join(base_path, "val/masks/*.png")))

    if not train_imgs: raise ValueError("Dataset paths are empty! Check your folder.")

    train_loader = DataLoader(NeuroScanDataset(train_imgs, train_masks), batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(NeuroScanDataset(val_imgs, val_masks), batch_size=CONFIG["batch_size"], pin_memory=True)

    # 2. Setup Training Objects
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = HybridFocalDiceLoss(lambda_dice=0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=CONFIG["patience_scheduler"])

    # FEATURE 3: Robust Resume Logic
    start_epoch, best_val_loss, history_list = 0, float('inf'), []
    if os.path.exists(CONFIG["checkpoint_path"]):
        print(f"🔄 Resuming from checkpoint and applying fine-tuning LR...")
        ckpt = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 🔥 CRITICAL CHANGE: Force override the optimizer's LR to 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = CONFIG["lr"]
            
        start_epoch, best_val_loss = ckpt['epoch'] + 1, ckpt['best_loss']
        if os.path.exists(CONFIG["log_path"]):
            history_list = pd.read_csv(CONFIG["log_path"]).to_dict(orient='records')

    # 4. Training Loop
    for epoch in range(start_epoch, CONFIG["total_epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['total_epochs']}")
        
        for images, masks in pbar:
            images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 5. Validation Phase
        model.eval()
        val_loss, val_dice, val_prec, val_rec, val_iou = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                
                # FEATURE 4: Modular Metrics & Visuals
                p, r, iou = calculate_metrics(outputs, masks)
                val_prec += p; val_rec += r; val_iou += iou
                
                inter = (torch.sigmoid(outputs) * masks).sum()
                union = torch.sigmoid(outputs).sum() + masks.sum()
                val_dice += (2. * inter + 1e-6) / (union + 1e-6)

                if i == 0: # Save visuals for the first batch
                    save_visual_comparison(images, masks, outputs, f"{CONFIG['plots_dir']}viz_ep{epoch+1}.png")
                    save_confusion_matrix(outputs, masks, f"{CONFIG['plots_dir']}cm_ep{epoch+1}.png")

        # Compile & Log Metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "dice": (val_dice / len(val_loader)).item(),
            "precision": val_prec / len(val_loader),
            "recall": val_rec / len(val_loader),
            "iou": val_iou / len(val_loader)
        }
        history_list.append(metrics)
        pd.DataFrame(history_list).to_csv(CONFIG["log_path"], index=False)
        save_training_plots(CONFIG["log_path"], CONFIG["plots_dir"])
        scheduler.step(metrics["val_loss"])

        # FEATURE 5: Best Model Saving
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_val_loss
            }, CONFIG["checkpoint_path"])
            print(f"⭐ Best Model Saved (Epoch {epoch+1})")

    print(f"✅ Full Scale Training Complete.")

if __name__ == "__main__":
    run_full_training()