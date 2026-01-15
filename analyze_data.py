import os
import cv2
import matplotlib
matplotlib.use('Agg') # Ye line GUI window khulne se rokegi (Error Fix)
import matplotlib.pyplot as plt
import random

SPLIT_DIR = "BraTS_Split"

def visualize_sample():
    # Train folder se kuch random images uthao
    train_img_dir = os.path.join(SPLIT_DIR, "train", "images")
    train_mask_dir = os.path.join(SPLIT_DIR, "train", "masks")
    
    # Check if folder exists
    if not os.path.exists(train_img_dir):
        print(f"❌ Error: {train_img_dir} nahi mila. Pehle split_data.py run karo.")
        return

    files = os.listdir(train_img_dir)
    
    # 5 Random samples pick karo
    samples = random.sample(files, 5)
    
    plt.figure(figsize=(15, 6)) # Thoda bada size
    
    for i, f in enumerate(samples):
        img_path = os.path.join(train_img_dir, f)
        mask_path = os.path.join(train_mask_dir, f)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Plot Image
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"MRI\n{f.split('_slice')[0]}", fontsize=8) # File name bhi dikhega
        plt.axis("off")
        
        # Plot Mask
        plt.subplot(2, 5, i+6)
        plt.imshow(mask, cmap="gray")
        plt.title("Tumor Mask", fontsize=8)
        plt.axis("off")
    
    plt.tight_layout()
    
    # IMPORTANT: Image ko save kar rahe hain taaki tum dekh sako
    save_path = "data_quality_check.png"
    plt.savefig(save_path)
    print(f"✅ Success! Image saved as '{save_path}'.")
    print("👉 VS Code ke file explorer mein 'data_quality_check.png' open karke check karo.")

if __name__ == "__main__":
    visualize_sample()