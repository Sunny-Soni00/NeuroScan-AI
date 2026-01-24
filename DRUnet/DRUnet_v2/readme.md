# 🧠 DRUnet V2: Attention-Guided 2.5D Brain Tumor Segmentation

> **A high-precision deep learning framework achieving 90.3% Dice Score on BraTS 2021 and 83% generalization on unseen BraTS 2019 data using pseudo-3D spatial context.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Dice Score](https://img.shields.io/badge/Dice-90.37%25-green)

---

## 🚀 Key Innovations
This project moves beyond standard 2D U-Nets by integrating **spatial depth** without the computational cost of full 3D models.

* **2.5D Spatial Input:** Stacks slices ($Z-1, Z, Z+1$) to provide volumetric context to a 2D architecture.
* **Attention-Guided Architecture:** Uses **Attention Gates** in the decoder to suppress background noise and focus on tumor regions.
* **Squeeze-and-Excitation (SE) Blocks:** dynamic channel recalibration to highlight relevant features.
* **Hybrid Loss Engine:** Combines **Focal Loss** (for hard-to-classify pixels) and **Dice Loss** (for segmentation overlap).
* **Inference Optimization:** Implements **Test-Time Augmentation (TTA)** for robust predictions.

---

## 📊 Performance Metrics

| Metric | Validation (BraTS 21) | Internal Test (Held-out) | External Test (BraTS 19) |
| :--- | :---: | :---: | :---: |
| **Dice Score** | **90.37%** | **87.69%** | **83.03%** |
| **Precision** | 92.69% | - | - |
| **Recall** | 89.66% | - | 82.73% |

> **Note:** The model demonstrates exceptional generalization, maintaining >83% accuracy on the completely unseen BraTS 2019 dataset without any fine-tuning.

---

## 🏗️ Architecture Overview

The model is a **Deep Residual U-Net (DRUnet)** with the following modifications:
1.  **Encoder:** Dilated Residual Blocks to expand receptive fields.
2.  **Bottleneck:** High-capacity feature extraction (1024 filters).
3.  **Decoder:** Equipped with Attention Gates to filter skip connections from the encoder.
4.  **Input:** 3-Channel tensor representing spatial depth ($256 \times 256 \times 3$).

---

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
pip install torch torchvision albumentations nibabel pandas opencv-python tqdm
2. Training
Bash
python train_drunet_v2.py
Configured for RTX 5060 (8GB VRAM) with Mixed Precision (AMP).

3. Evaluation
To run the master evaluation script (Plots + Internal Test + External Generalization):

Bash
python final_evaluation_v2.py
📂 Project Structure
DRUnet_v2/
├── dataset_balance_v2.py    # Custom 2.5D Dataloader (PNG/NIfTI)
├── model_drunet_v2.py       # Attention DRUnet Architecture
├── train_drunet_v2.py       # Training Loop with Scheduler & AMP
├── utils_v2.py              # Metrics & Hybrid Loss
├── final_evaluation_v2.py   # Master Audit Script
└── results/                 # Checkpoints and Logs
Author: Sunny Hardware: NVIDIA RTX 5060 (Blackwell)


---

### 📄 File 2: Detailed Project Report (Logic & Reasoning)
---

# 📑 Project Report: DRUnet V2 (Context-Aware Segmentation)

## 1. Problem Statement
Brain tumor segmentation is challenging due to the high variance in tumor shape, size, and location. Standard 2D CNNs treat each MRI slice independently, losing critical 3D spatial information. Conversely, full 3D networks require immense computational resources (24GB+ VRAM).

**Objective:** To bridge this gap by developing a **2.5D architecture** that captures volumetric context on consumer hardware (8GB VRAM) while achieving state-of-the-art accuracy.

---

## 2. Methodology & Logic

### 2.1 Data Strategy: The "Pseudo-3D" Approach
Instead of feeding a single grayscale image ($1 \times H \times W$) to the model, we engineered a **2.5D Input**:
* **Logic:** A radiologist doesn't look at one slice in isolation; they look at the previous and next slices to confirm a tumor boundary.
* **Implementation:** We stack three consecutive slices ($Z-1, Z, Z+1$) into the RGB channels of the input tensor.
* **Benefit:** The model learns continuity. If a spot appears in slice $Z$ but not in $Z-1$ or $Z+1$, the model learns it is likely noise, not a tumor.

### 2.2 Architecture: Why DRUnet V2?
We moved beyond the standard U-Net to a **Deep Residual U-Net (DRUnet)** with two critical enhancements:
1.  **Attention Gates (AGs):** Standard U-Nets pass noisy features through skip connections. AGs filter these signals, suppressing background activity and highlighting tumor regions before concatenation.
2.  **Squeeze-and-Excitation (SE) Blocks:** These blocks adaptively recalibrate channel-wise feature responses, effectively teaching the network "which feature map is more important" at any given step.

### 2.3 Training Dynamics
* **Loss Function:** We faced a massive class imbalance (99% healthy tissue vs. 1% tumor).
    * *Solution:* **Hybrid Focal Dice Loss**.
    * *Dice Loss:* Optimizes the global overlap (IoU).
    * *Focal Loss:* Applies a dynamic scaling factor to focus training on "hard" negatives (boundaries), preventing the model from just predicting "background" everywhere.
* **Optimization:**
    * Used **Mixed Precision (AMP)** to reduce VRAM usage by 40%, allowing a Batch Size of 16 on RTX 5060.
    * **The Saturation Fix:** At Epoch 10, the model plateaued at 89%. We manually dropped the Learning Rate from $10^{-4}$ to $10^{-5}$, enabling the model to converge to a precise local minimum (90.3%).

---

## 3. Results Analysis

### 3.1 Quantitative Performance
* **Validation Dice:** 90.37% (with TTA)
* **Internal Test (BraTS 21):** 87.69%
* **External Generalization (BraTS 19):** 83.03%

### 3.2 Key Observations
1.  **The "Stuck at 89%" Phenomenon:** Initially, the model hovered around 89% Dice. This was due to the learning rate being too aggressive for fine boundary delination. The LR Drop was the decisive factor in breaking this ceiling.
2.  **Test-Time Augmentation (TTA):** During inference, we averaged predictions from the original image, horizontal flip, and vertical flip. This "Free Lunch" strategy improved the Dice score by ~0.2% and significantly smoothed ragged prediction edges.
3.  **Generalization:** The model achieved **83% Dice** on the 2019 dataset without seeing a single image from it during training. This proves the model learned **anatomical features**, not just dataset biases.

---

## 4. Hardware Utilization
* **GPU:** NVIDIA RTX 5060 Laptop GPU.
* **Thermal Constraints:** Training pushed temps to 83°C. Efficient batching and AMP were crucial to preventing thermal throttling.
* **VRAM Efficiency:** Peak usage was 7.0 GB / 8.0 GB, maximizing hardware potential without OOM errors.

---

## 5. Conclusion
DRUnet V2 successfully demonstrates that **2.5D context** combined with **Attention mechanisms** can rival 3D performance benchmarks. The system is robust, precise (92.6% Precision), and computationally efficient, making it suitable for deployment in resource-constrained clinical environments.