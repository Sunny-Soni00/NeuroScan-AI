"""
Train MobileNetV2-UNet on BraTS 2021 (2.5D slices)
====================================================
Same dataset, loss, augmentation, and metrics as DRUNetv2 training
for a fair comparison.

Usage:
    python train_mobilenetv2.py                    # train from scratch
    python train_mobilenetv2.py --resume           # resume from checkpoint
    python train_mobilenetv2.py --epochs 30        # custom epoch count
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import albumentations as A
from tqdm import tqdm

# Import from DRUnet_v2 — same dataset & utils for fair comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DRUnet", "DRUnet_v2"))
from dataset_balance_v2 import BraTSDataset25D
from utils_v2 import (
    save_checkpoint,
    plot_comprehensive_metrics,
    calculate_all_metrics,
    HybridFocalDiceLoss,
)

from model_mobilenetv2 import MobileNetV2UNet

# ========================  CONFIG  ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4          # higher than DRUNetv2 (1e-5) because encoder is pretrained
BATCH_SIZE = 16
TOTAL_EPOCHS = 20
DATA_BASE_PATH = "../BraTS_Split"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Same augmentations as DRUNetv2
train_transform = A.Compose([
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.CLAHE(p=0.2),
])


# ========================  TRAIN  ========================
def train(args):
    print("=" * 65)
    print("  MobileNetV2-UNet  —  Brain Tumor Segmentation")
    print("=" * 65)
    print(f"  Device     : {DEVICE}")
    print(f"  LR         : {args.lr}")
    print(f"  Batch      : {args.batch_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Data       : {DATA_BASE_PATH}")
    print("=" * 65)

    # ---- Model ----
    model = MobileNetV2UNet(in_channels=3, out_channels=1, pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}  (DRUNetv2 has 33,003,053)")

    # ---- Optimiser & Scheduler ----
    # Differential LR: lower for pretrained encoder, higher for decoder
    encoder_params = list(model.enc1.parameters()) + list(model.enc2.parameters()) + \
                     list(model.enc3.parameters()) + list(model.enc4.parameters()) + \
                     list(model.enc5.parameters())
    decoder_params = list(model.up5.parameters()) + list(model.up4.parameters()) + \
                     list(model.up3.parameters()) + list(model.up2.parameters()) + \
                     list(model.up1.parameters()) + list(model.final.parameters())

    optimizer = optim.Adam([
        {"params": encoder_params, "lr": args.lr * 0.1},   # encoder: 1/10 LR
        {"params": decoder_params, "lr": args.lr},          # decoder: full LR
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)
    loss_fn = HybridFocalDiceLoss()
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    # ---- Resume ----
    start_epoch = 0
    checkpoint_path = os.path.join(RESULTS_DIR, "mobilenetv2_checkpoint.pth.tar")
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            start_epoch = ckpt.get("epoch", 0)
        else:
            model.load_state_dict(ckpt)
        print(f"  Resumed from epoch {start_epoch}")

    # ---- Data ----
    train_loader = DataLoader(
        BraTSDataset25D(os.path.join(DATA_BASE_PATH, "train"), transform=train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        BraTSDataset25D(os.path.join(DATA_BASE_PATH, "val")),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"  Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples\n")

    # ---- History ----
    history = {"train_loss": [], "val_loss": [], "dice": [], "iou": [], "precision": [], "recall": []}
    best_dice = 0.0

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)

            if scaler:
                with torch.amp.autocast("cuda"):
                    preds = model(x)
                    loss = loss_fn(preds, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x)
                loss = loss_fn(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # ---- Validation ----
        model.eval()
        val_loss = 0
        m_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                preds = model(x)
                val_loss += loss_fn(preds, y).item()
                m = calculate_all_metrics(preds, y)
                for k in m_sum:
                    m_sum[k] += m[k]

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        for k in m_sum:
            m_sum[k] /= len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        for k in m_sum:
            history[k].append(m_sum[k])

        scheduler.step(avg_val)

        # ---- Save ----
        is_best = m_sum["dice"] > best_dice
        if is_best:
            best_dice = m_sum["dice"]

        save_checkpoint(
            {"state_dict": model.state_dict(), "epoch": epoch + 1, "best_dice": best_dice},
            filename=checkpoint_path,
        )
        if is_best:
            save_checkpoint(
                model.state_dict(),
                filename=os.path.join(RESULTS_DIR, "mobilenetv2_best.pth.tar"),
            )

        pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "mobilenetv2_metrics.csv"), index=False)
        plot_comprehensive_metrics(history, save_path=RESULTS_DIR)

        print(
            f"  Ep {epoch+1:>2d} | Loss: {avg_train:.4f}/{avg_val:.4f} | "
            f"Dice: {m_sum['dice']:.4f} | IoU: {m_sum['iou']:.4f} | "
            f"Recall: {m_sum['recall']:.4f}"
            + ("  *best*" if is_best else "")
        )

    print(f"\n  Training complete. Best Dice: {best_dice:.4f}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Best model : results/mobilenetv2_best.pth.tar")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2-UNet on BraTS")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
