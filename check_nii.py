import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_nifti(nii_path, mask_path=None):
    # 1. Load the 3D NIfTI file
    img = nib.load(nii_path)
    data = img.get_fdata() # Convert to numpy array
    
    # 2. Pick the middle slice (usually where tumor is most visible)
    slice_idx = data.shape[2] // 2 
    
    plt.figure(figsize=(12, 6))

    # Plot the MRI Slice
    plt.subplot(1, 2, 1)
    plt.imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
    plt.title(f"MRI Slice: {slice_idx}")
    plt.axis('off')

    # 3. Load Mask if available to verify alignment
    if mask_path and os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
        plt.subplot(1, 2, 2)
        # Overlay: MRI in gray, Mask in red (using alpha for transparency)
        plt.imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        plt.imshow(mask[:, :, slice_idx].T, cmap='Reds', alpha=0.5, origin='lower')
        plt.title("Verification: MRI + Doctor's Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("nii_verification_check.png")
    print(f"✅ Verification image saved as 'nii_verification_check.png'")

# Example usage (Replace with your actual .nii file paths)
nii_file = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii"
mask_file = "/home/sunny/BrainTumor_AI/aryashah2k/brain-tumor-segmentation-brats-2019/versions/1/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_seg.nii"
visualize_nifti(nii_file, mask_file)