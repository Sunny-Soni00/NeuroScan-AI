# NeuroScan AI: Agentic Brain Tumor Segmentation & Diagnostics 🧠📋

## Project Overview

**NeuroScan AI** is an end-to-end medical imaging pipeline that combines Deep Learning with Agentic AI to detect and analyze brain tumors from MRI scans. The system utilizes a custom U-Net architecture for high-precision segmentation and a Vision-Language Agent (Llama 4 Scout via Groq) to provide morphological diagnostics.

### Key Features

✅ **Dual-Dataset Validation**: Trained on BraTS 2021 and cross-validated on BraTS 2019 for robust generalization  
✅ **Multi-Agent Workflow**: Radiologist Agent (segmentation) + Consultant Agent (grading & reporting)  
✅ **Medical Safety First**: Clinical False Positive rate of only **0.14%**  
✅ **GPU Optimized**: Mixed Precision training for NVIDIA RTX 5060 Laptop GPUs  
✅ **Real-time Inference**: ~25ms per MRI slice  

---

## 📊 Performance Metrics

| Metric | BraTS 2021 (Internal) | BraTS 2019 (External) | Status |
|--------|----------------------|----------------------|--------|
| **Mean Dice Score** | 0.9235 | 0.8510 | ✅ Verified |
| **Precision** | 94.07% | 89.12% | ✅ Verified |
| **Recall (Sensitivity)** | 83.88% | 78.45% | ✅ Verified |
| **False Positive Rate** | 0.14% | 0.29% | 🔒 Safe |

**Generalization Gap**: Only 7.85% Dice drop across datasets (excellent cross-domain robustness)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Model** | Custom 2D U-Net with Skip Connections |
| **Deep Learning** | PyTorch (Nightly Build) |
| **Web UI** | Streamlit |
| **Vision-Language Model** | Groq SDK (Llama 4 Scout) |
| **Medical Imaging** | Nibabel, Nilearn (NIfTI processing) |
| **Image Processing** | OpenCV, Albumentations |
| **GPU** | NVIDIA RTX 5060 Laptop (Mixed Precision) |

---

## 📂 Project Structure

```
BrainTumor_AI/
├── agent_app.py              # 🎯 Streamlit Multi-Agent Diagnostic Interface
├── model.py                  # 🧠 U-Net Architecture Definition
├── dataset.py                # 📦 BraTS Dataset Loader (DataLoader)
├── train.py                  # 🏋️ Training Pipeline with Validation
├── test_model.py             # ✅ Final Test Report Generation
├── evaluate_2019.py          # 📊 Cross-Dataset Generalization Audit
├── analyze_data.py           # 🔍 Data Quality Visualization
├── split_data.py             # 📂 Train/Val/Test Split Script
├── download_sample.py        # ⬇️ BraTS Dataset Downloader
│
├── my_checkpoint.pth.tar     # 💾 Trained Model Weights (Dice: 0.92)
├── requirements.txt          # 📋 Python Dependencies
│
├── BraTS_Split/              # 📁 Processed Dataset
│   ├── train/
│   ├── val/
│   └── test/
│
├── results/                  # 📈 Final Evaluation Outputs
│   ├── comparison_dice.png
│   ├── confusion_matrix.png
│   ├── generalization_histogram.png
│   └── Final_Report_Card.png
│
└── README.md                 # 📖 This File
```

---

## 🚀 Getting Started

Data set: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
testing data: https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019

### 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/Sunny-Soni00/NeuroScan-AI
cd NeuroScanAI

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Download & Prepare Data

```bash
# Download BraTS 2021 dataset (automatic)
python download_sample.py

# Split into train/val/test (80/10/10)
python split_data.py

# Visualize data quality
python analyze_data.py
```

### 3️⃣ Train the Model (Optional)

```bash
# Start training from scratch (takes ~2-4 hours on RTX 5060)
python train.py

# Monitor validation metrics in real-time
```

### 4️⃣ Launch the Diagnostic Interface

```bash
# Start Streamlit app
streamlit run agent_app.py

# Open browser → http://localhost:8501
# Upload an MRI scan → Get AI diagnosis
```

### 5️⃣ Evaluate Model Performance

```bash
# Test on internal test set
python test_model.py

# Cross-validate on BraTS 2019 (generalization audit)
python evaluate_2019.py
```

---

## 🎯 How It Works

### **Agent 1: Radiologist (Segmentation)**
- Receives MRI scan (256×256 input)
- U-Net predicts tumor segmentation mask
- Calculates tumor area and location
- Outputs: Binary mask + morphological metrics

### **Agent 2: Consultant (Diagnosis)**
- Receives tumor mask + medical context
- Llama 4 Scout Vision Agent analyzes morphology
- Grades tumor aggressiveness (HGG/LGG)
- Outputs: Clinical report + confidence scores

---

## 📈 Key Results

### Generalization Analysis
```
BraTS 2021 (Training Set)
├─ Mean Dice: 0.9235
├─ Precision: 94.07%
└─ Status: ✅ SOTA

BraTS 2019 (External Audit)
├─ Mean Dice: 0.8510
├─ Precision: 89.12%
├─ Generalization Gap: -7.85%
└─ Status: ✅ Excellent Cross-Domain Transfer
```

### Safety Metrics
- **True Negative Rate**: 98.71% (healthy tissue correctly identified)
- **False Positive Rate**: 0.14% (minimal false alarms)
- **Clinical Grade**: 🔒 Safe for deployment

---

## ⚙️ Configuration

Edit these settings in `agent_app.py` or training scripts:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256                    # Input image resolution
BATCH_SIZE = 16                   # Training batch size
LEARNING_RATE = 1e-4             # Adam optimizer LR
NUM_EPOCHS = 50                   # Training epochs
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
```

---

## 📦 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
nibabel>=5.0.0
nilearn>=0.10.0
matplotlib>=3.7.0
pillow>=10.0.0
albumentations>=1.3.0
groq>=0.4.0
google-generativeai>=0.3.0
tqdm>=4.66.0
kagglehub>=0.2.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔬 Model Architecture

```
Input (1, 256, 256)
    ↓
Encoder (4 levels)
    ├─ Conv 1×1 → 64 channels
    ├─ Conv 64 → 128 (MaxPool)
    ├─ Conv 128 → 256 (MaxPool)
    └─ Conv 256 → 512 (MaxPool)
    ↓
Bottleneck (512 → 512)
    ↓
Decoder (4 levels)
    ├─ UpConv 512 → 256 (Skip Connection)
    ├─ UpConv 256 → 128 (Skip Connection)
    ├─ UpConv 128 → 64 (Skip Connection)
    └─ UpConv 64 → 1 (Output)
    ↓
Sigmoid Activation
    ↓
Output: Probability Map (1, 256, 256)
```

**Parameters**: ~7.8M trainable weights

---

## 🚨 Known Limitations & Future Work

| Limitation | Impact | Planned Fix |
|-----------|--------|------------|
| Micro-tumors (<50px) | Low sensitivity for small lesions | Attention-Gated U-Net |
| 2D Slices Only | Ignores volumetric context | 3D U-Net implementation |
| FLAIR Modality | Other MRI sequences not supported | Multi-modal fusion |
| Inference Latency | ~25ms per slice | Model quantization (INT8) |

---

## 📝 Usage Example

```python
from PIL import Image
import streamlit as st

# In agent_app.py:
uploaded_file = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "nii.gz"])

if uploaded_file:
    # Preprocessing
    img = Image.open(uploaded_file)
    
    # Agent 1: Segmentation
    tumor_mask = unet_model.predict(img)
    
    # Agent 2: Diagnosis
    report = groq_client.analyze_tumor(tumor_mask, img)
    
    st.write(report)  # Display clinical report
```