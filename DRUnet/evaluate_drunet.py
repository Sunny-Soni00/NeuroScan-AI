# import torch
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from model_drunet import DRUNet
# from dataset_balance import BraTSDataset
# from torch.utils.data import DataLoader
# import os

# # Results directory
# RESULTS_DIR = "results/evaluation"
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def run_safe_evaluation():
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
    
#     # Load the best weights from your high-capacity run
#     checkpoint = torch.load("results/drunet_highcap_best.pth.tar")
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()

#     DATA_PATH = "/home/sunny/BrainTumor_AI/BraTS2021_Raw"
#     all_patients = sorted(os.listdir(DATA_PATH))
#     val_patients = all_patients[40:50] 
#     val_ds = BraTSDataset(root_dir=DATA_PATH, patient_list=val_patients)
#     val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

#     # Initialize Confusion Matrix variables (Pixel-wise)
#     tp, tn, fp, fn = 0, 0, 0, 0
    
#     print(f"🔬 Evaluating {len(val_patients)} patients slice-by-slice to save RAM...")

#     with torch.no_grad():
#         for x, y in val_loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             output = torch.sigmoid(model(x))
#             preds = (output > 0.5).float()

#             # Calculating components without giant concatenations
#             tp += ((preds == 1) & (y == 1)).sum().item()
#             tn += ((preds == 0) & (y == 0)).sum().item()
#             fp += ((preds == 1) & (y == 0)).sum().item()
#             fn += ((preds == 0) & (y == 1)).sum().item()

#     # Medical Metrics Calculation
#     dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0

#     print(f"\n📊 Final Metrics Summary:")
#     print(f"Dice Score: {dice:.4f}")
#     print(f"Sensitivity (Recall): {sensitivity:.4f}")
#     print(f"Specificity: {specificity:.4f}")
#     print(f"Precision: {precision:.4f}")

#     # --- Plotting Confusion Matrix ---
#     cm = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title("Pixel-wise Confusion Matrix")
#     plt.xlabel("Predicted Labels")
#     plt.ylabel("Actual Labels")
#     plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    
#     print(f"📈 Confusion Matrix saved in {RESULTS_DIR}")

# if __name__ == "__main__":
#     run_safe_evaluation()

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_drunet import DRUNet
from dataset_balance import BraTSDataset
from torch.utils.data import DataLoader
import os

RESULTS_DIR = "results/final_evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_safe_evaluation():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # Loading the latest checkpoint (Epoch 10 weights)
    checkpoint = torch.load("results/drunet_highcap_best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    DATA_PATH = "/home/sunny/BrainTumor_AI/BraTS2021_Raw"
    all_patients = sorted(os.listdir(DATA_PATH))
    # Using 10% Validation set
    val_patients = all_patients[40:50] 
    val_ds = BraTSDataset(root_dir=DATA_PATH, patient_list=val_patients)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Statistical Accumulators (Streaming Logic)
    tp, tn, fp, fn = 0, 0, 0, 0
    
    print(f"🔬 Safe Evaluation: Processing {len(val_patients)} patients...")

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                output = torch.sigmoid(model(x))
            preds = (output > 0.5).float()

            # Accumulate counts without keeping giant arrays in RAM
            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    # Medical Metrics Calculation
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n✅ Final Stats After 10 Epochs:")
    print(f"Dice Score: {dice:.4f} | Recall: {sensitivity:.4f} | Specificity: {specificity:.4f}")

    # Plot Confusion Matrix
    cm = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Pixel-wise Confusion Matrix (Epoch 10)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(RESULTS_DIR, "final_confusion_matrix.png"))
    
    print(f"📈 Confusion Matrix saved in {RESULTS_DIR}")

if __name__ == "__main__":
    run_safe_evaluation()