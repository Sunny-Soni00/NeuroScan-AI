# NeuroScan AI: Brain Tumor Segmentation on Edge Devices

A comparative study of two deep learning architectures for brain tumor segmentation, trained on BraTS 2021 and deployed on NVIDIA Jetson Nano 4GB using TensorRT FP16.

## Results Summary

| Metric | DRUNetv2 | MobileNetV2-UNet |
|---|---|---|
| Parameters | 33.0M | **2.6M** (12.6x smaller) |
| Val Dice (Training) | **0.9037** | 0.8986 |
| Test Dice (Jetson TRT FP16) | 0.8127 | **0.8390** |
| FPS (Jetson Nano) | 30.4 | **134.5** (4.4x faster) |
| Latency | 34.1 ms | **8.6 ms** |
| ONNX Size | 126 MB | **10.2 MB** |
| TRT Engine Size | 63.9 MB | ~5 MB |
| Jetson RAM Usage | 519 MB | **408 MB** |

Both models maintain accuracy after FP16 quantisation (< 0.003 Dice difference).

## Approach

**2.5D Preprocessing**: For each target slice at position Z, we stack it with neighbours Z-1 and Z+1 to form a 3-channel input. This provides volumetric context without the memory cost of full 3D processing.

**Loss Function**: Hybrid Focal-Dice Loss combining Focal Loss (handles class imbalance, as tumors occupy <5% of pixels) and Dice Loss (directly optimises the overlap metric).

**Dataset**: BraTS 2021 (1251 volumes), split into 65,300 training / 8,293 validation / ~8,000 test 2D slices at 256x256 resolution.

## Models

### DRUNetv2 (Attention Deep Residual U-Net)
Custom architecture with Dilated Residual Blocks, Squeeze-and-Excitation channel attention, and Attention Gates on skip connections. 33M parameters, trained from scratch.

### MobileNetV2-UNet
ImageNet-pretrained MobileNetV2 encoder with a UNet-style decoder and skip connections. 2.6M parameters. Uses differential learning rate (encoder 0.1x, decoder 1x).

Both trained with identical settings: Adam optimiser, batch size 16, 20 epochs, same augmentations (rotation, flip, elastic transform, CLAHE), mixed precision (FP16).

## Project Structure

```
BrainTumor_AI/
│
├── DRUnet/                          # DRUNet v1 (baseline model)
│   ├── model_drunet.py              # Standard U-Net architecture
│   ├── train_drunet.py              # Training script
│   ├── dataset_balance.py           # 2D dataset loader
│   └── results/                     # Evaluation results & plots
│
├── DRUnet_v2/                       # DRUNetv2 (attention-guided, 2.5D)
│   ├── model_drunet_v2.py           # AttentionDRUNet architecture (33M params)
│   ├── train_drunet_v2.py           # Training with mixed precision
│   ├── dataset_balance_v2.py        # 2.5D dataset loader (Z-1, Z, Z+1)
│   ├── utils_v2.py                  # HybridFocalDiceLoss, metrics
│   ├── evaluate_with_tta.py         # Test-time augmentation evaluation
│   └── results/                     # Metrics, plots, reports
│
├── MobileNetV2_Seg/                 # MobileNetV2-UNet (lightweight)
│   ├── model_mobilenetv2.py         # MobileNetV2 encoder + UNet decoder (2.6M params)
│   ├── train_mobilenetv2.py         # Training with differential LR
│   ├── export_onnx.py               # ONNX export with weight inlining
│   └── results/                     # Metrics, laptop inference results
│
├── DRUnet_v2_jetson_deploy/         # Jetson deployment (DRUNetv2)
│   ├── convert_to_trt.py            # ONNX → TensorRT conversion
│   ├── run_inference_trt.py         # TensorRT inference + visualisation
│   ├── run_inference.py             # ONNX Runtime alternative
│   ├── verify_results.py            # Results analysis
│   └── test_data/                   # 30 test samples (PNGs + masks)
│
├── mobilenetv2_jetson_deploy/       # Jetson deployment (MobileNetV2)
│   ├── mobilenetv2_jetson.onnx      # Trained model (10.2 MB)
│   ├── convert_to_trt.py            # ONNX → TensorRT conversion
│   ├── run_inference_trt.py         # TensorRT inference + visualisation
│   ├── run_inference.py             # ONNX Runtime alternative
│   ├── verify_results.py            # Results analysis
│   └── test_data/                   # 30 test samples (PNGs + masks)
│
├── agent_app.py                     # Streamlit multi-agent diagnostic app
├── streamlit_drunetv2_proper.py     # DRUNetv2 Streamlit interface
├── streamlit_drunetv2_app.py        # Streamlit app (alternate)
├── test_drunetv2.py                 # DRUNetv2 test evaluation script
└── readme.md                        # This file
```

## Jetson Nano Deployment

Each model has a self-contained deploy folder. Copy to Jetson and run 3 commands:

```bash
# DRUNetv2 (need to provide your own ONNX, not in repo due to 126 MB size)
cd DRUnet_v2_jetson_deploy
python3 convert_to_trt.py --fp16
python3 run_inference_trt.py

# MobileNetV2 (ONNX included in repo, 10.2 MB)
cd mobilenetv2_jetson_deploy
python3 convert_to_trt.py --onnx mobilenetv2_jetson.onnx --fp16
python3 run_inference_trt.py
```

Visualisation output for each test image: `Input | Ground Truth | Prediction | Overlap` with TP (green), FP (red), FN (blue) colour coding.

## Dataset

- **Training**: [BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
- **Cross-validation**: [BraTS 2019](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

## Requirements

```
torch >= 2.0
torchvision
albumentations
opencv-python
numpy
pandas
tqdm
onnx
onnxruntime
```

## Hardware

- **Training**: NVIDIA RTX 5060 Laptop GPU (8 GB VRAM), mixed precision
- **Inference**: NVIDIA Jetson Nano 4 GB, TensorRT FP16, JetPack

## Author

Sunny Soni
