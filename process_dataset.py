import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
# Jahan data unzip hua hai
# (Note: Agar unzip ke baad folder ka naam alag ho, toh ise change kar lena)
DATASET_PATH = "BraTS2021_Raw" 
OUTPUT_PATH = "BraTS_Processed_2D"

# Output folders create karo
os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)

def normalize_image(img):
    """Image ko black & white (0-255) format mein convert karta hai"""
    img = np.nan_to_num(img)  # NaN errors hatao
    if np.max(img) != 0:
        img = img / np.max(img)  # 0 se 1 ke beech lao
        img = img * 255          # 0 se 255 ke beech lao
    return img.astype(np.uint8)

def process_dataset():
    # Check karein ki dataset folder hai ya nahi
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Folder '{DATASET_PATH}' nahi mila!")
        return

    # Kabhi kabhi zip ke andar ek aur folder hota hai, use handle karte hain
    # Agar BraTS2021_Raw ke andar seedhe folders nahi hain, toh andar dhundo
    possible_inner_folder = os.path.join(DATASET_PATH, "BraTS2021_Training_Data")
    if os.path.exists(possible_inner_folder):
        search_path = possible_inner_folder
    else:
        search_path = DATASET_PATH

    print(f"📂 Scanning data in: {search_path}")

    # Sare patient folders list karo
    subjects = [f for f in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, f))]
    print(f"Found {len(subjects)} patients. Processing start... (Isme time lagega)")

    count = 0
    
    # Har patient ko process karo
    for subject in tqdm(subjects):
        subject_path = os.path.join(search_path, subject)
        
        # Files ke naam standard hote hain
        flair_file = os.path.join(subject_path, f"{subject}_flair.nii.gz") # MRI Image
        seg_file = os.path.join(subject_path, f"{subject}_seg.nii.gz")     # Tumor Mask

        # Sirf tab process karo jab dono files maujood hon
        if os.path.exists(flair_file) and os.path.exists(seg_file):
            try:
                # 3D Data Load karo
                flair_vol = nib.load(flair_file).get_fdata()
                seg_vol = nib.load(seg_file).get_fdata()

                # Har slice (layer) ko check karo
                # Depth usually 155 hoti hai
                n_slices = flair_vol.shape[2]
                
                for i in range(n_slices):
                    mask_slice = seg_vol[:, :, i]
                    
                    # LOGIC: Sirf wahi images save karo jisme Tumor dikh raha ho
                    # (Khali images save karke disk bharna bekar hai)
                    if np.max(mask_slice) > 0:
                        img_slice = flair_vol[:, :, i]

                        # Image Normalize karo
                        img_final = normalize_image(img_slice)
                        
                        # Mask ko visible banao (0=Black, 255=White)
                        mask_final = (mask_slice > 0).astype(np.uint8) * 255

                        # Save as PNG
                        fname = f"{subject}_slice_{i}.png"
                        cv2.imwrite(os.path.join(OUTPUT_PATH, "images", fname), img_final)
                        cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", fname), mask_final)
                        count += 1
            except Exception as e:
                print(f"Skipping {subject}: Error {e}")
    
    print(f"✅ Processing Done! Total {count} tumor images saved in '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    process_dataset()