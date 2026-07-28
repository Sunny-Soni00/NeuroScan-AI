import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
from tqdm import tqdm
from model_drunet_v2 import AttentionDRUNet

# ================= ⚙️ CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model_v2.pth"
# UPDATE THESE IN full_model_audit.py IF NEEDED
TEST_DATA_DIR = "test_data" # Your folder name
OUTPUT_FOLDER = "Final_Audit_Results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/visual_samples", exist_ok=True)

# ================= 🧠 LOAD MODEL =================
def load_neuroscan_model():
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ================= 📈 METRICS ENGINE =================
def calculate_metrics(pred_prob, target, threshold=0.5):
    pred = (pred_prob > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    return dice, recall, tp, fp, fn, tn

# ================= 🚀 EXECUTION LOOP =================
def run_audit():
    model = load_neuroscan_model()
    images = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "images", "*.png")))
    if not images:
        images = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.png")))
    masks = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "masks/*.png")))

    if not images:
        raise FileNotFoundError(f"No test images found under '{TEST_DATA_DIR}'")
    if not masks:
        raise FileNotFoundError(f"No test masks found under '{TEST_DATA_DIR}/masks'")
    if len(images) != len(masks):
        raise ValueError(
            f"Image/mask count mismatch: {len(images)} images vs {len(masks)} masks"
        )
    
    results = {"dice": [], "recall": []}
    px = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


    print(f"🔬 Auditing {len(images)} slices on {DEVICE}...")

    for i in tqdm(range(len(images))):
        # 1. Pre-process (Z-Score)
        img = cv2.imread(images[i]).astype(np.float32)
        mean, std = np.mean(img), np.std(img)
        img_norm = (img - mean) / (std + 1e-8)
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # 2. Reflexive Padding
        h, w = img_t.shape[2], img_t.shape[3]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        img_padded = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')

        # 3. Inference
        with torch.no_grad():
            out = torch.sigmoid(model(img_padded))
            out = out[:, :, :h, :w] 

        # 4. Metrics
        d, r, tp, fp, fn, tn = calculate_metrics(out, mask_t)
        results["dice"].append(d)
        results["recall"].append(r)
        px["tp"] += tp; px["fp"] += fp; px["fn"] += fn; px["tn"] += tn

        # --- UPDATED VISUAL SAMPLE LOGIC ---
        # Saving every 5th slice since total count is 30
        if i % 5 == 0:
            # 1. Prepare Input MRI (reload original for display consistency)
            # Use images[i] which points to the current PNG being audited
            # Read middle channel for display (assuming standard v2 3-channel input)
            orig_img = cv2.imread(images[i])
            if orig_img is None:
                orig_img = np.zeros((h, w, 3), dtype=np.uint8) # Fallback
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

            # 2. Prepare Ground Truth Mask
            # Convert mask (float32, 0-1) loaded earlier back to uint8 (0-255)
            gt_viz = (mask * 255).astype(np.uint8)

            # 3. Prepare Binary Prediction Mask
            # Convert tensor 'out' to numpy, threshold at 0.5, scale to uint8 (0-255)
            pred_np = out[0, 0].cpu().numpy()
            pred_viz = (pred_np > 0.5).astype(np.uint8) * 255

            # 4. Create Side-by-Side Staged Image (Input | GT | Pred)
            # Create a larger canvas to hold all three images vertically stacked or horizontally staged.
            # Staging horizontally for better report layout.
            # Add titles using Matplotlib for clear identification.
            
            plt.figure(figsize=(15, 5))
            
            # Input subplot
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img, cmap='gray')
            plt.title(f'Input Slice (Z={i})', fontsize=12)
            plt.axis('off')

            # Ground Truth subplot
            plt.subplot(1, 3, 2)
            plt.imshow(gt_viz, cmap='gray')
            plt.title('Ground Truth', fontsize=12)
            plt.axis('off')

            # Prediction subplot (add current Dice score for context)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_viz, cmap='gray')
            plt.title(f'Prediction (Dice: {d:.2f})', fontsize=12, color='green' if d > 0.8 else 'red')
            plt.axis('off')

            plt.tight_layout()
            
            # Save the plotted figure to Final_Audit_Results/visual_samples/
            sample_fn = os.path.join(OUTPUT_FOLDER, "visual_samples", f"slice_{i}_comparison.png")
            plt.savefig(sample_fn, dpi=100, bbox_inches='tight')
            plt.close() # Close figure to free memory

    # Save summary report
    with open(f"{OUTPUT_FOLDER}/audit_summary.txt", "w") as f:
        f.write(f"Final Dice: {np.mean(results['dice'])*100:.2f}%\n")
        f.write(f"Final Recall: {np.mean(results['recall'])*100:.2f}%\n")

    print(f"✅ Laptop Audit complete. Results in '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    run_audit()