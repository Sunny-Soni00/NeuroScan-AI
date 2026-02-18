# DRUNetv2 — Jetson Nano Deployment

## What's in this folder

```
jetson_deploy/
├── drunetv2_jetson.onnx       # Trained model (126 MB)
├── convert_to_trt.py          # Step 1: Convert ONNX → TensorRT
├── run_inference_trt.py       # Step 2: Run inference (TensorRT)
├── run_inference.py           # Alternative: Run with ONNX Runtime
├── verify_results.py          # Step 3: Check results
├── test_data/
│   ├── npz/  (30 files)       # 2.5D input stacks [prev, cur, next slice]
│   ├── masks/ (30 files)      # Ground truth segmentation masks
│   └── *.png (30 files)       # Preview images
└── README.md                  # This file
```

## Setup (one time)

```bash
# 1. Copy this entire folder to Jetson Nano
#    USB drive, SCP, or any method — just get the whole folder there

# 2. On Jetson, set max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# 3. Check dependencies (should already be on JetPack)
python3 -c "import numpy, cv2; print('OK')"
```

## Run (3 commands)

```bash
cd ~/jetson_deploy

# STEP 1: Convert model to TensorRT (one time, takes 2-5 min)
python3 convert_to_trt.py --fp16

# STEP 2: Run inference on all 30 test images
python3 run_inference_trt.py

# STEP 3: Check results
python3 verify_results.py
```

That's it. Results appear in `results/combined/` — each image shows:

```
Input | Ground Truth | Prediction | Overlap
```

With Dice, IoU, Precision, Recall, FPS printed below each image.

## If TensorRT conversion fails

Some older JetPack versions may not support opset 18. Use ONNX Runtime instead:

```bash
python3 run_inference.py
```

Same results, just slower (~3-5 FPS vs ~15-25 FPS with TRT).

## Expected output

- **Dice score**: ~0.90+ (median ~0.935)
- **FPS (FP16 TRT)**: ~15-25
- **FPS (ONNX Runtime)**: ~3-5
- **Results folder**: `results/combined/*.png` + `results/metrics.json`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Slow FPS | `sudo nvpmodel -m 0 && sudo jetson_clocks` |
| Out of memory | Close other apps, reboot, try again |
| `libcudart.so` not found | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| No GPU provider | Install onnxruntime-gpu: `pip3 install onnxruntime-gpu` |
| jtop not found | `sudo pip3 install jetson-stats` (optional, for monitoring) |
