import os
import torch
import torch.nn as nn
import torch.optim as optim
from model_drunet import DRUNet
from utils import save_checkpoint, load_checkpoint, check_accuracy, plot_learning_curves
from dataset_balance import BraTSDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= ⚙️ RESUME CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
START_EPOCH = 5   # Where we left off
NUM_EPOCHS = 10    # Target total epochs
LOAD_MODEL = True # Set to True to continue training
DATA_PATH = "/home/sunny/BrainTumor_AI/BraTS2021_Raw"
CHECKPOINT_FILE = "results/drunet_highcap_best.pth.tar"

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        BCE = nn.functional.binary_cross_entropy_with_logits(
            inputs.view(-1), targets.view(-1), reduction='mean'
        )
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(probs.sum() + targets.sum() + smooth)  
        return BCE + dice_loss

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float().unsqueeze(1)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return epoch_loss / len(loader)

def main():
    model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') 

    # --- 🔄 RESUME LOGIC ---
    if LOAD_MODEL and os.path.exists(CHECKPOINT_FILE):
        print(f"🔄 Resuming from checkpoint: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint["state_dict"])
        # Optional: Load optimizer state if saved in checkpoint
        # optimizer.load_state_dict(checkpoint["optimizer"]) 
    
    # --- DATA LOADING ---
    all_patients = sorted(os.listdir(DATA_PATH))
    train_patients, val_patients = all_patients[:40], all_patients[40:50] 

    train_ds = BraTSDataset(root_dir=DATA_PATH, patient_list=train_patients)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = BraTSDataset(root_dir=DATA_PATH, patient_list=val_patients)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_losses, val_dice_scores = [], []
    best_dice = 0.4315 # Our best from Epoch 4

    print(f"🏁 Continuing training from Epoch {START_EPOCH+1} to {NUM_EPOCHS}...")
    
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(avg_loss)
        
        current_dice = check_accuracy(val_loader, model, device=DEVICE).item()
        val_dice_scores.append(current_dice)

        plot_learning_curves(train_losses, val_dice_scores)
        
        if current_dice > best_dice:
            best_dice = current_dice
            save_checkpoint({"state_dict": model.state_dict()}, filename="drunet_highcap_best.pth.tar")
            print(f"✨ New Best Dice: {best_dice:.4f}")
        
        print(f"📈 Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {current_dice:.4f}")

if __name__ == "__main__":
    main()