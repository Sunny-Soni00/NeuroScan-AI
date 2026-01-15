import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir):
        """
        Ye function tab chalta hai jab hum Dataset create karte hain.
        Ye bas list banata hai ki files kahan rakhi hain.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        
        # Saari files ke naam list kar lete hain
        self.images = os.listdir(self.image_dir)
        print(f"✅ Dataset Loaded: {len(self.images)} images found.")

    def __len__(self):
        """GPU ko batata hai ki total kitni images hain"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Ye sabse important part hai! 
        Jab GPU maangta hai: "Mujhe 5th image do", tab ye function chalta hai.
        """
        img_name = self.images[idx]
        
        # 1. Disk se Image aur Mask load karo
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Grayscale mein read karte hain (0 = Black, 255 = White)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. Safety Check (Kabhi kabhi file corrupt ho sakti hai)
        if image is None or mask is None:
            # Agar file kharab hai, toh bas ek khali image bhej do (Crash rokne ke liye)
            return torch.zeros((1, 256, 256)), torch.zeros((1, 256, 256))

        # 3. Preprocessing (Data ko 0 se 1 ke beech lao - AI ko ye pasand hai)
        image = image / 255.0
        mask = mask / 255.0

        # 4. Resize (Sab images ka size same hona chahiye - 256x256)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # 5. Convert to Tensor (PyTorch format)
        # Hame dimension badhana padega: (Height, Width) -> (Channels, Height, Width)
        image = np.expand_dims(image, axis=0) # Ab ye (1, 256, 256) ban gaya
        mask = np.expand_dims(mask, axis=0)

        # Final return: Float32 format mein (GPU ke liye best)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# --- Chhota Test Block ---
if __name__ == "__main__":
    # Ye sirf tab chalega jab hum 'python dataset.py' run karenge check karne ke liye
    ds = BrainTumorDataset("BraTS_Processed_2D")
    img, mask = ds[0] # Pehli image maang kar dekhte hain
    print(f"Test Success! Image Shape: {img.shape}, Mask Shape: {mask.shape}")