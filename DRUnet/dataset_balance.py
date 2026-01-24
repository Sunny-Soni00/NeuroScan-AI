import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2

class BraTSDataset(Dataset):
    """
    Professional Medical Dataset Loader.
    Handles NIfTI file processing and Intensity Normalization.
    """
    def __init__(self, root_dir, patient_list, transform=None):
        self.root_dir = root_dir
        self.patients = patient_list
        self.transform = transform
        self.slices = []

        # Logic to index all slices and identify which contain tumors
        # Based on EDA: 40.9% are tumor-positive
        for patient in self.patients:
            patient_path = os.path.join(self.root_dir, patient)
            flair_path = os.path.join(patient_path, f"{patient}_flair.nii.gz")
            seg_path = os.path.join(patient_path, f"{patient}_seg.nii.gz")
            
            if os.path.exists(flair_path) and os.path.exists(seg_path):
                # We store indices and a 'has_tumor' flag for the Weighted Sampler
                seg_data = nib.load(seg_path).get_fdata()
                for i in range(seg_data.shape[2]):
                    has_tumor = np.sum(seg_data[:, :, i] > 0) > 0
                    self.slices.append((flair_path, seg_path, i, has_tumor))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        flair_p, seg_p, slice_idx, _ = self.slices[index]
        
        # Load specific 3D slice
        flair_img = nib.load(flair_p).get_fdata()[:, :, slice_idx]
        seg_img = nib.load(seg_p).get_fdata()[:, :, slice_idx]

        # 1. Resize to 256x256
        image = cv2.resize(flair_img, (256, 256))
        mask = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 2. Z-Score Intensity Normalization (Data Science Standard)
        # Formula: $Z = \frac{x - \mu}{\sigma}$
        if np.std(image) > 0:
            image = (image - np.mean(image)) / np.std(image)
        
        # 3. Binary classification of mask (All tumor labels to 1)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            # Apply Augmentations here
            pass

        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32)