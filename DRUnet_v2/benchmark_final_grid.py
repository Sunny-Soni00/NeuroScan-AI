import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob
import json
from tqdm import tqdm

# Internal Imports
from model_drunet_v2 import AttentionDRUNet 
from losses import HybridFocalDiceLoss 

# --- HARDWARE OPTIMIZATION SETTINGS ---
# Enables the built-in cudnn auto-tuner to find the best algorithm for your hardware
torch.backends.cudnn.benchmark = True 

# --- DATA PREPARATION LOGIC ---
def get_subset_paths(base_path, subset_fraction=0.05):
    """
    Retrieves a subset of image and mask paths for benchmarking.
    Default is 5% to balance evidence quality and computation time.
    """
    img_dir = os.path.join(base_path, "train/images") 
    mask_dir = os.path.join(base_path, "train/masks")
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    num_samples = max(1, int(len(img_paths) * subset_fraction)) if img_paths else 0
    return img_paths[:num_samples], mask_paths[:num_samples]

# --- OPTIMIZED TRAINING FUNCTION ---
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, total_epochs):
    """
    Executes one training epoch with real-time progress tracking.
    Uses non_blocking transfers and set_to_none for VRAM efficiency.
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_idx+1}/{total_epochs}", leave=False)
    
    for i, (images, masks) in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # set_to_none=True provides a modest speed boost by reducing memory overhead
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
    return running_loss / len(loader)

# --- VALIDATION FUNCTION ---
def validate(model, loader, device):
    """
    Evaluates the model using the Dice Similarity Coefficient.
    Ensures memory is not consumed by gradients during evaluation.
    """
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(images))
            
            intersection = (outputs * masks).sum()
            union = outputs.sum() + masks.sum()
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.item())
            
    return np.mean(dice_scores)

# --- RAM CACHED DATASET CLASS ---
class RAMCachedDataset(torch.utils.data.Dataset):
    """
    Pre-loads processed slices into system RAM to bypass Disk I/O bottlenecks.
    Applies Z-Score normalization during the loading phase.
    """
    def __init__(self, img_paths, mask_paths):
        self.images, self.masks = [], []
        print(f"📥 Caching {len(img_paths)} slices into System RAM...")
        for img_p, m_p in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="Caching"):
            img = cv2.imread(img_p).astype(np.float32)
            # Z-Score Normalization: (x - mean) / std
            img = (img - np.mean(img)) / (np.std(img) + 1e-8)
            
            mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            
            self.images.append(torch.from_numpy(img).permute(2, 0, 1))
            self.masks.append(torch.from_numpy(mask).unsqueeze(0))
            
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i], self.masks[i]

# --- MAIN GRID SEARCH EXECUTION ---
def run_final_grid_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")
    
    # Path configuration logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.abspath(os.path.join(script_dir, "..", "BraTS_Split"))
    train_imgs, train_masks = get_subset_paths(base_data_path, subset_fraction=0.05)

    if not train_imgs:
        raise ValueError(f"Dataset Not Found: Verify path {base_data_path}")
    
    # Batch Size 4 is chosen to prevent Shared GPU Memory swapping on 8GB VRAM
    train_ds = RAMCachedDataset(train_imgs, train_masks)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, pin_memory=True, num_workers=2)
    
    lrs = [1e-3, 1e-4, 1e-5] 
    wds = [1e-2, 1e-4]
    total_epochs = 3 
    
    results_dir = "BENCHMARK_RESULTS"
    os.makedirs(results_dir, exist_ok=True)
    full_results = {}

    for lr in lrs:
        for wd in wds:
            combo_name = f"LR_{lr}_WD_{wd}"
            history_file = os.path.join(results_dir, f"{combo_name}_history.json")
            model_file = os.path.join(results_dir, f"{combo_name}_model.pth")
            
            # Resume/Skip logic to handle interruptions
            if os.path.exists(history_file):
                print(f"⏭️ Configuration {combo_name} found in cache. Skipping...")
                with open(history_file, 'r') as f:
                    full_results[combo_name] = json.load(f)
                continue

            print(f"\n🔥 TESTING HYPERPARAMETERS: {combo_name}")
            model = AttentionDRUNet(in_channels=3, out_channels=1).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = HybridFocalDiceLoss(lambda_dice=0.7, gamma_focal=2.0)
            
            history = {'loss': [], 'dice': []}
            
            for epoch in range(total_epochs):
                avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs)
                avg_dice = validate(model, train_loader, device)
                
                history['loss'].append(avg_loss)
                history['dice'].append(avg_dice)
                print(f"   📈 Epoch {epoch+1}: Loss = {avg_loss:.4f} | Dice = {avg_dice:.4f}")

                # Checkpoint saving for stability
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }
                torch.save(checkpoint, model_file)

            # Record final results for this specific configuration
            with open(history_file, 'w') as f:
                json.dump(history, f)
            full_results[combo_name] = history

    # --- GENERATING FINAL EVIDENCE GRID ---
    print("\n📊 Generating Final Research Grid...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    for i, lr in enumerate(lrs):
        for j, wd in enumerate(wds):
            config_name = f"LR_{lr}_WD_{wd}"
            data = full_results[config_name]
            
            ax = axes[i, j]
            ax.plot(data['loss'], 'r-o', label='Training Loss')
            ax_secondary = ax.twinx()
            ax_secondary.plot(data['dice'], 'b-s', label='Dice Coefficient')
            
            ax.set_title(f"Configuration: {config_name}", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            if j == 0: ax.set_ylabel("Loss Magnitude")
            if j == 1: ax_secondary.set_ylabel("Segmentation Accuracy (Dice)")
            
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "final_experimental_grid.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Benchmark Complete. Logical evidence saved to {plot_path}")

if __name__ == "__main__":
    run_final_grid_search()