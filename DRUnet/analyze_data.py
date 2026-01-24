import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# DATASET PATH
DATA_PATH = "/home/sunny/BrainTumor_AI/BraTS2021_Raw" # Update strictly

def analyze_dataset():
    tumor_sizes = []
    slice_coverage = [] # Kitni slices mein tumor hai vs empty slices
    
    print("🔬 Starting Deep Data Analysis...")
    
    patients = sorted(os.listdir(DATA_PATH))
    # Taking sample of 50 patients for quick statistical significance
    # (Full dataset would take 1 hour, sample is enough for distribution logic)
    for patient in tqdm(patients[:50]):
        seg_path = os.path.join(DATA_PATH, patient, f"{patient}_seg.nii.gz")
        if not os.path.exists(seg_path): continue
        
        seg_data = nib.load(seg_path).get_fdata()
        
        # 1. Tumor Size Analysis (Non-zero pixels per slice)
        for i in range(seg_data.shape[2]):
            slice_mask = seg_data[:, :, i]
            tumor_pixels = np.sum(slice_mask > 0)
            
            if tumor_pixels > 0:
                tumor_sizes.append(tumor_pixels)
                slice_coverage.append(1) # Tumor Present
            else:
                slice_coverage.append(0) # Empty Slice

    # --- PLOTTING THE EVIDENCE ---
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Tumor Size Distribution (Log Scale)
    # Hume dekhna hai ki kya chote tumors ki sankhya zyada hai?
    sns.histplot(tumor_sizes, bins=50, kde=True, ax=ax[0], color='purple', log_scale=True)
    ax[0].set_title("Why DRUNet? Tumor Size Variance (Log Scale)")
    ax[0].set_xlabel("Tumor Area (Pixels)")
    ax[0].set_ylabel("Frequency")
    ax[0].text(0.5, 0.9, "Wide spread indicates need for Multi-Scale features (Dilation)", 
               transform=ax[0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Plot 2: Class Imbalance
    tumor_slices = sum(slice_coverage)
    empty_slices = len(slice_coverage) - tumor_slices
    ax[1].pie([tumor_slices, empty_slices], labels=['Tumor Slices', 'Empty Slices'], 
              autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], explode=(0.1, 0))
    ax[1].set_title("Class Imbalance: Tumor vs Background")
    
    plt.tight_layout()
    plt.savefig("data_analysis_report.png")
    print("✅ Analysis Report Saved: 'data_analysis_report.png'")
    print(f"Stats: Smallest Tumor: {min(tumor_sizes)} px | Largest Tumor: {max(tumor_sizes)} px")

if __name__ == "__main__":
    analyze_dataset()