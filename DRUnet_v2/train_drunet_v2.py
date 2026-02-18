import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import albumentations as A
from tqdm import tqdm

from dataset_balance_v2 import BraTSDataset25D
from model_drunet_v2 import AttentionDRUNet
from utils_v2 import save_checkpoint, plot_comprehensive_metrics, calculate_all_metrics, HybridFocalDiceLoss

# ================= ⚙️ V2.1 CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
BATCH_SIZE = 16 
START_EPOCH = 0
TOTAL_EPOCHS = 20 
RESUME = True # Set to True to load epoch 5 checkpoint
DATA_BASE_PATH = "../../BraTS_Split" 
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

train_transform = A.Compose([
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.CLAHE(p=0.2),
])

def train_v2_resume():
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
    loss_fn = HybridFocalDiceLoss()
    scaler = torch.amp.GradScaler('cuda')

    # --- 🔄 RESUME LOGIC ---
    checkpoint_path = os.path.join(RESULTS_DIR, "v2_checkpoint.pth.tar")
    if RESUME and os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint) # In your current utils, state_dict is saved directly
        print("✅ Successfully resumed from previous state.")

        global START_EPOCH
        START_EPOCH = 13

    train_loader = DataLoader(BraTSDataset25D(os.path.join(DATA_BASE_PATH, "train"), transform=train_transform), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(BraTSDataset25D(os.path.join(DATA_BASE_PATH, "val")), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # History loading for CSV logging
    history = {"train_loss": [], "val_loss": [], "dice": [], "iou": [], "precision": [], "recall": []}

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            with torch.amp.autocast('cuda'):
                preds = model(x)
                loss = loss_fn(preds, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        v_loss = 0
        m_acc = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                preds = model(x)
                v_loss += loss_fn(preds, y).item()
                m = calculate_all_metrics(preds, y)
                for k in m_acc: m_acc[k] += m[k]

        # Stats calculation
        avg_train = train_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        for k in m_acc: history[k].append(m_acc[k] / len(val_loader))

        # Save to CSV for Excel analysis [NEW FEATURE]
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(RESULTS_DIR, "v2_metrics_log.csv"), index=False)

        scheduler.step(avg_val)
        plot_comprehensive_metrics(history, save_path=RESULTS_DIR)
        save_checkpoint(model.state_dict(), filename=os.path.join(RESULTS_DIR, "v2_checkpoint.pth.tar"))
        
        print(f"📈 Ep {epoch+1} | Dice: {history['dice'][-1]:.4f} | Recall: {history['recall'][-1]:.4f}")

if __name__ == "__main__":
    train_v2_resume()