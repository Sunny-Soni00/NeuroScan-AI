import matplotlib.pyplot as plt
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Data from Phase 1 (Epoch 1-5) and Phase 2 (Epoch 6-10)
epochs = list(range(1, 11))
# Loss values from terminal logs
train_loss = [0.9650, 0.5313, 0.2524, 0.1433, 0.1084, 0.1079, 0.0848, 0.0727, 0.0655, 0.0651]
# Dice values from terminal logs
val_dice = [0.3149, 0.4125, 0.3850, 0.4315, 0.4235, 0.4235, 0.4205, 0.4053, 0.4156, 0.4099]

def plot_master_results():
    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss (Convergence)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
    plt.title('DRUNet Loss Convergence (Epoch 1-10)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot 2: Segmentation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice, 'g-s', label='Validation Dice')
    plt.axhline(y=0.9235, color='r', linestyle='--', label='UNet Baseline (0.92)')
    plt.title('Segmentation Accuracy (Epoch 1-10)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/master_learning_curves.png")
    print("✅ Master Plot saved: results/master_learning_curves.png")

if __name__ == "__main__":
    plot_master_results()