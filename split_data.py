import os
import shutil
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
DATA_PATH = "BraTS_Processed_2D"
IMAGES_DIR = os.path.join(DATA_PATH, "images")
MASKS_DIR = os.path.join(DATA_PATH, "masks")

# Naya structure kaisa dikhega
OUTPUT_DIR = "BraTS_Split"
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}

def create_split():
    # 1. Pehle saare Patients ki list nikalo (Images ki nahi)
    all_files = os.listdir(IMAGES_DIR)
    
    # Filename format: BraTS2021_00000_slice_55.png
    # Hume bas "BraTS2021_00000" chahiye (Patient ID)
    patient_ids = set()
    for f in all_files:
        if f.endswith(".png"):
            pid = "_".join(f.split("_")[:2]) # BraTS2021_00000
            patient_ids.add(pid)
    
    patient_ids = list(patient_ids)
    random.seed(42) # Seed set kar rahe hain taaki har baar same split ho (Reproducibility)
    random.shuffle(patient_ids)
    
    total_patients = len(patient_ids)
    print(f"👨‍⚕️ Total Unique Patients: {total_patients}")

    # 2. Patients ko divide karo
    train_count = int(total_patients * SPLIT_RATIO["train"])
    val_count = int(total_patients * SPLIT_RATIO["val"])
    
    train_patients = patient_ids[:train_count]
    val_patients = patient_ids[train_count : train_count + val_count]
    test_patients = patient_ids[train_count + val_count:]

    print(f"📊 Split Stats -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

    # 3. Files move karo
    sets = {"train": train_patients, "val": val_patients, "test": test_patients}
    
    for split_name, patients in sets.items():
        # Folders banao
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "masks"), exist_ok=True)
        
        print(f"🚀 Moving files for {split_name} set...")
        
        # Set ke patients ko set mein convert karo for fast lookup
        patient_set = set(patients)

        # Saari files check karo, agar wo patient is set mein hai toh move karo
        for f in tqdm(all_files):
            # Check ID
            pid = "_".join(f.split("_")[:2])
            
            if pid in patient_set:
                # Source Paths
                src_img = os.path.join(IMAGES_DIR, f)
                src_mask = os.path.join(MASKS_DIR, f)
                
                # Destination Paths
                dst_img = os.path.join(OUTPUT_DIR, split_name, "images", f)
                dst_mask = os.path.join(OUTPUT_DIR, split_name, "masks", f)
                
                # Copy (Move nahi karenge safety ke liye, Copy karenge)
                shutil.copy(src_img, dst_img)
                shutil.copy(src_mask, dst_mask)

    print("✅ Data Split Complete! Folder Structure: 'BraTS_Split/' created.")

if __name__ == "__main__":
    create_split()