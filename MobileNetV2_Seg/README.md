# MobileNetV2-UNet: Lightweight Brain Tumor Segmentation

MobileNetV2-UNet for brain tumor segmentation on BraTS 2021, designed as a lightweight alternative to DRUNetv2 for edge deployment.

## Architecture

- **Encoder**: Pretrained MobileNetV2 (ImageNet) — 5 stages producing features at 128x128 (16ch), 64x64 (24ch), 32x32 (32ch), 16x16 (96ch), 8x8 (320ch)
- **Decoder**: UNet-style with bilinear upsampling, skip connections, and double convolution blocks
- **Output**: 1-channel logits at 256x256, sigmoid applied during inference
- **Parameters**: 2,614,513 (12.6x smaller than DRUNetv2's 33M)

## Training

```bash
python train_mobilenetv2.py                 # train from scratch (20 epochs)
python train_mobilenetv2.py --resume        # resume from checkpoint
python train_mobilenetv2.py --epochs 30     # custom epoch count
```

Uses differential learning rate: encoder (pretrained) at 0.1x LR, decoder at full LR.  
Same dataset, loss function, augmentations, and metrics as DRUNetv2 for fair comparison.

### Training Results

| Epoch | Val Dice | Val IoU | Val Loss |
|---|---|---|---|
| 1 | 0.8696 | 0.7848 | 0.1569 |
| 10 | 0.8941 | 0.8209 | 0.1223 |
| 18 (best) | **0.8986** | **0.8269** | 0.1176 |
| 20 | 0.8977 | 0.8265 | 0.1184 |

## ONNX Export

```bash
python export_onnx.py                       # exports mobilenetv2_jetson.onnx (10.2 MB)
```

Produces a single-file ONNX with weights inlined (no external data files).  
Verification: 100% mask agreement between PyTorch and ONNX.

## Jetson Deployment

See `mobilenetv2_jetson_deploy/` for the self-contained deployment folder.

```bash
cd ../mobilenetv2_jetson_deploy
python3 convert_to_trt.py --onnx mobilenetv2_jetson.onnx --fp16
python3 run_inference_trt.py
```

## Files

| File | Description |
|---|---|
| `model_mobilenetv2.py` | MobileNetV2-UNet architecture definition |
| `train_mobilenetv2.py` | Training script (imports dataset/loss from DRUnet_v2) |
| `export_onnx.py` | ONNX export with weight inlining and verification |
| `results/comprehensive_metrics.png` | Training curves (loss, dice, precision, recall) |
| `results/mobilenetv2_metrics.csv` | Per-epoch metrics log |
| `results/laptop_test/` | ONNX inference results on 30 test samples |
