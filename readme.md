# NeuroScan AI: Brain Tumor Segmentation on Edge Devices

NeuroScan AI is a comparative study of two deep learning architectures for
brain tumor segmentation. Both models are trained on the BraTS 2021 dataset,
cross-validated on BraTS 2019, and deployed on an NVIDIA Jetson Nano 4 GB using
TensorRT FP16. The repository contains the full training and evaluation code,
the Jetson deployment pipeline for each model, a set of Streamlit inference
apps, and an interactive web-based Explainer that visualises the internal
behaviour of the networks stage by stage.

## Table of Contents

- [Highlights](#highlights)
- [Results Summary](#results-summary)
- [Method](#method)
- [Model Architectures](#model-architectures)
- [Model Distribution](#model-distribution)
- [Interactive Explainer](#interactive-explainer)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Jetson Nano Deployment](#jetson-nano-deployment)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Hardware](#hardware)
- [Author](#author)

## Highlights

- Two architectures trained under identical protocols for a fair comparison:
  a custom attention-guided residual U-Net (DRUNetv2) and a lightweight,
  ImageNet-pretrained MobileNetV2-UNet.
- End-to-end edge deployment: PyTorch to ONNX to TensorRT FP16 engine, with a
  three-command deploy folder for each model.
- Accuracy preserved after FP16 quantisation (less than 0.003 Dice difference).
- The lightweight model runs at 134.5 FPS on a Jetson Nano while matching the
  accuracy of the 12.6x larger model on held-out test data.
- An interactive Explainer (see [Interactive Explainer](#interactive-explainer))
  that opens the "black box": it renders the feature maps, attention maps,
  Squeeze-and-Excitation gates and probability distributions for a real MRI
  slice, and lets the user step through the forward pass one stage at a time.

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

## Method

### 2.5D Preprocessing

For each target slice at axial position Z, the slice is stacked with its two
immediate neighbours, Z-1 and Z+1, to form a 3-channel input of shape
`3 x 256 x 256`. This gives the network volumetric context around the slice of
interest without the memory and compute cost of a full 3D model. The middle
channel is always the slice being segmented; the outer channels act as context.
Intensities are normalised to the range 0.0 to 1.0.

### Loss Function

Training uses a Hybrid Focal-Dice Loss. The Focal Loss term addresses the
severe class imbalance, since tumour tissue occupies less than 5 percent of the
pixels in a typical slice, while the Dice Loss term directly optimises the
overlap metric that the models are evaluated on. The implementation lives in
`DRUnet_v2/utils_v2.py` (`HybridFocalDiceLoss`).

### Dataset and Splits

BraTS 2021 contains 1251 annotated volumes. After 2.5D slice extraction and
filtering, the data is split into approximately 65,300 training slices, 8,293
validation slices, and roughly 8,000 held-out test slices, all at 256x256
resolution. BraTS 2019 is used as an independent cross-validation set to measure
generalisation to a different acquisition distribution.

### Training Configuration

Both models are trained with identical settings so that the comparison reflects
architecture rather than tuning:

- Optimiser: Adam
- Batch size: 16
- Epochs: 20
- Precision: mixed precision (FP16) with automatic loss scaling
- Augmentations: rotation, horizontal and vertical flips, elastic transform,
  and CLAHE
- MobileNetV2-UNet additionally uses a differential learning rate: the
  pretrained encoder is trained at 0.1x the base rate, the decoder at 1x.

## Model Architectures

### DRUNetv2 (Attention Deep Residual U-Net)

A custom architecture trained from scratch, defined in
`DRUnet_v2/model_drunet_v2.py`. It combines three ideas on top of a standard
U-Net:

- **Dilated Residual Blocks** at every encoder and decoder stage: two dilated
  3x3 convolutions with BatchNorm and ReLU, wrapped in a residual connection.
- **Squeeze-and-Excitation** channel attention inside each block: a global
  average pool followed by a small two-layer bottleneck that produces a
  per-channel gain in the range 0 to 1, rescaling the feature map.
- **Attention Gates** on the skip connections: before an encoder feature map is
  concatenated into the decoder, it is multiplied by a learned 0-to-1
  coefficient map that suppresses irrelevant regions.

The bottleneck uses dilation 2, giving each unit a wide receptive field over the
16x16 feature map without additional parameters. Approximately 33M parameters.

### MobileNetV2-UNet

A lightweight model defined in `MobileNetV2_Seg/model_mobilenetv2.py`. It pairs
an ImageNet-pretrained MobileNetV2 encoder (inverted residual blocks with
depthwise-separable convolutions) with a compact U-Net decoder that upsamples
bilinearly, concatenates the encoder skip connection, and applies two 3x3
convolutions per stage. There are no SE blocks and no attention gates; the
decoder recovers all spatial detail from the raw skips. Approximately 2.6M
parameters, trained with a differential learning rate.

## Model Distribution

The repository includes the ready-to-run MobileNetV2-UNet ONNX model:

- [`mobilenetv2_jetson.onnx`](mobilenetv2_jetson_deploy/mobilenetv2_jetson.onnx) - 2.5D input (`3 x 256 x 256`), binary segmentation output, approximately 10.2 MB.

It can be used with ONNX Runtime or converted to a TensorRT FP16 engine using the
instructions in [Jetson Nano Deployment](#jetson-nano-deployment). The DRUNetv2
PyTorch checkpoints and ONNX export are not included as tracked release assets;
the training and export code remains in the repository.

For a professional model release, the recommended free distribution target is a
Hugging Face model repository with a model card, license, preprocessing details,
evaluation results, and checksums. No Hugging Face repository has been published
yet because this account is not authenticated in the current environment. The
tracked ONNX file above remains directly available from GitHub in the meantime.

## Interactive Explainer

`visualizer/` contains a self-contained web application that visualises what
happens inside the two segmentation networks. It is inspired by the Poloclub
Diffusion Explainer: instead of presenting the model as "image in, mask out",
it lays the entire network out as a horizontal pipeline and lets the user step
through the forward pass one stage at a time, inspecting the intermediate
tensors that are normally hidden.

### What it shows

- A horizontal **architecture map** of the whole network, from the 2.5D input
  through the encoder, the bottleneck, the attention gates, the decoder, and
  the head. A playhead travels along the pipeline while each block lights up as
  the signal reaches it.
- For every encoder, bottleneck and decoder stage: a montage of individual
  feature-map channels, a peak-activation summary, and, for DRUNetv2, the
  Squeeze-and-Excitation channel-gain bar chart for that slice.
- For each DRUNetv2 attention gate: the psi coefficient map blended over the
  input slice, showing where the model chooses to pass skip information through.
- For the head: the sigmoid probability map, a probability histogram, and a
  live threshold slider that recomputes the binary mask, the true-positive /
  false-positive / false-negative overlay, and the Dice score in the browser.
- For every stage: the input and output tensor shapes, the parameter count, and
  an ordered list of the operations that block performs.

The application also has a light and dark theme, keyboard controls, and encodes
the current model, slice and stage in the URL so a view can be shared as a link.

### Running the Explainer

The Explainer is static HTML, CSS and JavaScript with no build step. Serve the
`visualizer/` directory with any static file server:

```bash
cd visualizer
python -m http.server 8777
```

Then open `http://localhost:8777/` in a browser. The site root redirects to the
application at `visualizer/web/`.

Controls: the play and pause button steps automatically through the stages with
a speed slider; the left and right arrow keys step manually; the spacebar
toggles play. Click any block, or the attention diamonds, to jump directly to
that stage. Feature-map images enlarge on click.

### Regenerating the assets

The feature maps, attention maps and probability images are precomputed offline
by running the models on the Jetson deployment test slices with forward hooks.
The generated files live under `visualizer/data/` and are not committed to the
repository. To recreate them:

```bash
python visualizer/extract/export_activations.py            # a few sample slices
python visualizer/extract/export_activations.py --limit 30  # all deploy test slices
```

This reads the slices in `DRUnet_v2_jetson_deploy/test_data/npz/`, runs both
models, and writes `visualizer/data/<model>/<slice>/` plus a top-level
`visualizer/data/manifest.json` that the front-end consumes.

### Adding another model

1. Add an entry to `MODELS` in `visualizer/extract/config.py` with an ordered
   list of stages. Each stage names a hookable submodule path (see
   `model.named_modules()`), its input and output shapes, its operation list,
   and a plain-language caption.
2. Add a `build_<name>()` function in
   `visualizer/extract/export_activations.py` and register it in `BUILDERS`.
3. Re-run the exporter. The front-end picks up any model present in
   `manifest.json` automatically and draws the correct pipeline shape, including
   attention gates only where the architecture has them.

The baseline `DRUNet` in `DRUnet/model_drunet.py` is the obvious third model to
add.

## Repository Structure

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
│   ├── convert_to_trt.py            # ONNX to TensorRT conversion
│   ├── run_inference_trt.py         # TensorRT inference + visualisation
│   ├── run_inference.py             # ONNX Runtime alternative
│   ├── verify_results.py            # Results analysis
│   └── test_data/                   # 30 test samples (PNGs + masks)
│
├── mobilenetv2_jetson_deploy/       # Jetson deployment (MobileNetV2)
│   ├── mobilenetv2_jetson.onnx      # Trained model (10.2 MB)
│   ├── convert_to_trt.py            # ONNX to TensorRT conversion
│   ├── run_inference_trt.py         # TensorRT inference + visualisation
│   ├── run_inference.py             # ONNX Runtime alternative
│   ├── verify_results.py            # Results analysis
│   └── test_data/                   # 30 test samples (PNGs + masks)
│
├── visualizer/                      # Interactive Explainer (static web app)
│   ├── extract/                     # Offline: forward hooks and asset export
│   │   ├── config.py                # Per-model stage definitions and captions
│   │   ├── hooks.py                 # Forward-hook recorder and tensor rendering
│   │   └── export_activations.py    # Runs the models, writes visualizer/data/
│   ├── data/                        # Generated assets (git-ignored)
│   └── web/                         # index.html, css/, js/ (no build step)
│
├── agent_app.py                     # Streamlit multi-agent diagnostic app
├── streamlit_common_models_app.py   # Streamlit app: run multiple models on one image
├── streamlit_drunetv2_proper.py     # DRUNetv2 Streamlit interface
├── streamlit_drunetv2_app.py        # Streamlit app (alternate)
├── test_drunetv2.py                 # DRUNetv2 test evaluation script
└── readme.md                        # This file
```

## Getting Started

### Environment

Create a virtual environment and install the dependencies listed under
[Requirements](#requirements):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision albumentations opencv-python numpy pandas tqdm onnx onnxruntime streamlit
```

### Running the Streamlit apps

```bash
streamlit run streamlit_common_models_app.py   # compare models on an uploaded image
streamlit run streamlit_drunetv2_proper.py     # DRUNetv2 single-model interface
```

### Running the Explainer

```bash
cd visualizer
python -m http.server 8777
# open http://localhost:8777/
```

If the generated assets are not present, create them first with
`python visualizer/extract/export_activations.py --limit 30` from the repository
root. See [Interactive Explainer](#interactive-explainer) for details.

### Training

```bash
python DRUnet_v2/train_drunet_v2.py
python MobileNetV2_Seg/train_mobilenetv2.py
```

### Evaluation

```bash
python DRUnet_v2/evaluate_with_tta.py          # test-time augmentation evaluation
python test_drunetv2.py                         # test-set evaluation
```

## Jetson Nano Deployment

Each model has a self-contained deploy folder. Copy it to the Jetson and run
three commands:

```bash
# DRUNetv2 (provide your own ONNX; not in the repo due to its 126 MB size)
cd DRUnet_v2_jetson_deploy
python3 convert_to_trt.py --fp16
python3 run_inference_trt.py

# MobileNetV2 (ONNX included in the repo, 10.2 MB)
cd mobilenetv2_jetson_deploy
python3 convert_to_trt.py --onnx mobilenetv2_jetson.onnx --fp16
python3 run_inference_trt.py
```

The visualisation output for each test image is a strip of
`Input | Ground Truth | Prediction | Overlap`, with true positives in green,
false positives in red, and false negatives in blue.

## Datasets

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

The Streamlit apps additionally require `streamlit`. The Explainer front-end has
no dependencies beyond a static file server; its asset-export script uses the
same PyTorch environment as training, plus `matplotlib` and `Pillow` for
rendering.

## Hardware

- **Training**: NVIDIA RTX 5060 Laptop GPU (8 GB VRAM), mixed precision
- **Inference**: NVIDIA Jetson Nano 4 GB, TensorRT FP16, JetPack

## Author

Sunny Soni
