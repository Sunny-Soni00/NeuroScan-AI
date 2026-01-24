import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset25D(Dataset):
    def __init__(self, folder_path, img_size=256, transform=None):
        self.img_dir = os.path.join(folder_path, "images")
        self.mask_dir = os.path.join(folder_path, "masks")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = self.img_files[idx]
        # Logic to find neighbor slices based on filename
        prefix = "_".join(filename.split("_")[:-1]) 
        current_idx = int(filename.split("_")[-1].split(".")[0]) 

        def load_img(s_idx):
            path = os.path.join(self.img_dir, f"{prefix}_{s_idx}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                return cv2.resize(img, (self.img_size, self.img_size))
            return None

        # 2.5D Spatial Context
        img_z = load_img(current_idx)
        img_prev = load_img(current_idx - 1)
        img_next = load_img(current_idx + 1)

        if img_prev is None: img_prev = img_z
        if img_next is None: img_next = img_z

        img_stack = np.stack([img_prev, img_z, img_next], axis=-1) 

        mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img_stack, mask=mask)
            img_stack, mask = augmented['image'], augmented['mask']

        img_stack = (img_stack / 255.0).astype(np.float32)
        img_stack = np.transpose(img_stack, (2, 0, 1)) 

        return torch.tensor(img_stack), torch.tensor(mask)