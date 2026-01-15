import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNET
from dataset import BrainTumorDataset
from torch.utils.data import DataLoader
import os

# ================= CONFIGURATION =================
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16    # Reduce to 8 if you get "Out of Memory" error
NUM_EPOCHS = 5     # Training for 5 rounds
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

# New Paths after splitting
TRAIN_IMG_DIR = "BraTS_Split/train"
VAL_IMG_DIR = "BraTS_Split/val"

def check_accuracy(loader, model, device="cuda"):
    """
    Calculates the 'Dice Score' to measure how well the model predicts the tumor.
    Score 0 = Terrible, Score 1 = Perfect.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # Switch to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Predict
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # Convert probabilities to 0 or 1
            
            # Compare prediction with actual mask
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # Dice Score Formula: (2 * Intersection) / (Area A + Area B)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"✅ Accuracy Check: Got {num_correct}/{num_pixels} with Dice Score: {dice_score/len(loader):.4f}")
    model.train() # Switch back to training mode

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    The main training loop.
    """
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().to(DEVICE)

        # Forward Pass (using Mixed Precision for RTX 5060 speed)
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward Pass (updating weights)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

def main():
    print(f"🚀 Starting Training on {DEVICE} (RTX 5060)...")
    
    # Initialize Model
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') # For Mixed Precision

    # Load Data
    train_ds = BrainTumorDataset(TRAIN_IMG_DIR)
    val_ds = BrainTumorDataset(VAL_IMG_DIR)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY, 
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY, 
        shuffle=False,
    )

    # Start Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n📢 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # 1. Train on data
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # 2. Check accuracy on validation data
        check_accuracy(val_loader, model, DEVICE)
        
        # 3. Save Model Checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "my_checkpoint.pth.tar")
        print("💾 Model Saved!")

if __name__ == "__main__":
    main()