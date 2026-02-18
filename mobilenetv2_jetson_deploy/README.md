# MobileNetV2-UNet — Jetson Nano Deployment

Brain tumor segmentation using MobileNetV2-UNet (2.6M params) on Jetson Nano 4 GB.
Same 2.5D preprocessing and evaluation as DRUNetv2 for fair comparison.

## Quick Start (3 commands)

```bash
# 1. Convert ONNX → TensorRT FP16
python convert_to_trt.py --onnx mobilenetv2_jetson.onnx --fp16

# 2. Run inference on 30 test samples
python run_inference_trt.py

# 3. (Optional) Verify results
python verify_results.py
```

## Model Comparison

| Metric         | DRUNetv2      | MobileNetV2-UNet |
|----------------|---------------|------------------|
| Parameters     | 33.0M         | 2.6M (12.6× smaller) |
| ONNX Size      | 126 MB        | 10.2 MB          |
| Training Dice  | 0.9037        | 0.8986           |
| Test Mean Dice | 0.8125        | 0.8369           |

## Files

- `mobilenetv2_jetson.onnx` — ONNX model (10.2 MB, single file)
- `convert_to_trt.py` — ONNX → TensorRT conversion
- `run_inference_trt.py` — TensorRT inference with visualisations
- `run_inference.py` — ONNX Runtime inference (alternative)
- `verify_results.py` — Results analysis
- `test_data/` — 30 test samples (npz + masks + preview PNGs)
