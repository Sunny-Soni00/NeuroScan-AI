import zipfile
import os
from tqdm import tqdm

# Folder configuration
extract_to = "BraTS2021_Raw"
os.makedirs(extract_to, exist_ok=True)

# Find the zip file automatically
zip_files = [f for f in os.listdir() if f.endswith('.zip')]

if not zip_files:
    print("❌ No .zip file found! Check if it's in the download folder.")
else:
    zip_path = zip_files[0] # Take the first zip file
    print(f"📂 Found Zip: {zip_path}")
    print(f"🚀 Extracting to: {extract_to}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files for progress bar
        file_list = zip_ref.namelist()
        
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)
            
    print("✅ Extraction Complete! Check the 'BraTS2021_Raw' folder.")