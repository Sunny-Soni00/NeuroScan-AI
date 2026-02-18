#!/usr/bin/env python3
"""
MobileNetV2-UNet ONNX Runtime Inference for Jetson Nano
=======================================================
Alternative to TensorRT inference — uses ONNX Runtime with CUDA provider.

Loads proper 2.5D stacks from npz files (exported by export_test_data.py)
and produces Input | GT | Pred | Overlap visualisations.

Usage:
    python run_inference.py
    python run_inference.py --model mobilenetv2_jetson.onnx --data test_data --output results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    print("  pip install onnxruntime-gpu   (Jetson)")
    print("  pip install onnxruntime       (CPU)")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from jtop import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False

# ========================  CONFIG  ========================
IMG_SIZE = 256
THRESHOLD = 0.5


# ========================  DATA LOADING  ========================
def load_test_data(data_dir: str):
    """
    Load test data.  Prefers npz/ 2.5D stacks, falls back to PNG.
    Returns list of (input_stack [3,H,W], mask [H,W], filename).
    """
    npz_dir = os.path.join(data_dir, "npz")
    samples = []

    if os.path.isdir(npz_dir):
        for p in sorted(Path(npz_dir).glob("*.npz")):
            d = np.load(str(p), allow_pickle=True)
            samples.append((
                d["input"].astype(np.float32),
                d["mask"].astype(np.float32) if "mask" in d else None,
                str(d["filename"]),
            ))
        if samples:
            print(f"  Loaded {len(samples)} 2.5D stacks (npz/)")
            return samples

    # Fallback: PNG (replicate same slice 3×)
    mask_dir = os.path.join(data_dir, "masks")
    for p in sorted(Path(data_dir).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        inp = np.stack([img, img, img], axis=0)

        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mpath = os.path.join(mask_dir, p.name)
        if os.path.exists(mpath):
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)
        samples.append((inp, mask, p.name))

    print(f"  Loaded {len(samples)} images (PNG fallback — no real 2.5D context)")
    return samples


# ========================  METRICS  ========================
def compute_metrics(pred_prob, gt):
    pred = (pred_prob > THRESHOLD).astype(np.float32)
    gt = gt.astype(np.float32)
    tp = np.sum(pred * gt)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    tn = np.sum((1 - pred) * (1 - gt))
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    return {"dice": float(dice), "iou": float(iou), "precision": float(prec),
            "recall": float(rec), "accuracy": float(acc)}


# ========================  VISUALISATION  ========================
def create_combined_vis(input_stack, gt_mask, pred_prob, metrics):
    """
    Input | Ground Truth | Prediction | Overlap
    Overlap: green=TP, red=FP, blue=FN
    """
    H, W = IMG_SIZE, IMG_SIZE
    cur = input_stack[1]  # middle slice
    cur_u8 = (cur * 255).clip(0, 255).astype(np.uint8)
    inp_bgr = cv2.cvtColor(cur_u8, cv2.COLOR_GRAY2BGR)

    gt_bin = (gt_mask > 0.5).astype(np.uint8) if gt_mask is not None else np.zeros((H, W), np.uint8)
    pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)

    gt_vis = np.zeros((H, W, 3), np.uint8)
    gt_vis[:, :, 1] = gt_bin * 255

    pred_vis = np.zeros((H, W, 3), np.uint8)
    pred_vis[:, :, 2] = pred_bin * 255

    overlap = inp_bgr.copy()
    tp = (pred_bin & gt_bin).astype(bool)
    fp = (pred_bin & ~gt_bin).astype(bool)
    fn = (~pred_bin & gt_bin).astype(bool)
    a = 0.55
    overlap[tp] = (a * np.array([0, 255, 0]) + (1 - a) * overlap[tp]).astype(np.uint8)
    overlap[fp] = (a * np.array([0, 0, 255]) + (1 - a) * overlap[fp]).astype(np.uint8)
    overlap[fn] = (a * np.array([255, 0, 0]) + (1 - a) * overlap[fn]).astype(np.uint8)

    combined = np.hstack([inp_bgr, gt_vis, pred_vis, overlap])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Input",   (10,       25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "GT",      (W + 10,   25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "Pred",    (2*W + 10, 25), font, 0.6, (0, 0, 255), 2)
    cv2.putText(combined, "Overlap", (3*W + 10, 25), font, 0.6, (255, 255, 255), 2)

    bar = np.zeros((30, 4 * W, 3), np.uint8)
    txt = (f"Dice:{metrics['dice']:.3f} | IoU:{metrics['iou']:.3f} | "
           f"Prec:{metrics['precision']:.3f} | Rec:{metrics['recall']:.3f} | "
           f"FPS:{metrics.get('fps', 0):.1f}")
    cv2.putText(bar, txt, (10, 22), font, 0.5, (220, 220, 220), 1)
    combined = np.vstack([combined, bar])

    legend = np.zeros((22, 4 * W, 3), np.uint8)
    cv2.putText(legend, "Green=TP  Red=FP  Blue=FN", (10, 16), font, 0.45, (160, 160, 160), 1)
    return np.vstack([combined, legend])


# ========================  MAIN  ========================
def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 ONNX inference")
    parser.add_argument("--model", default="mobilenetv2_jetson.onnx")
    parser.add_argument("--data", default="test_data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}"); sys.exit(1)

    print("\n" + "=" * 70)
    print("  MobileNetV2-UNet ONNX Runtime Inference — Jetson Nano")
    print("=" * 70)

    # ---- Create ONNX session (force GPU) ----
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB
            "arena_extend_strategy": "kSameAsRequested",
        }),
        "CPUExecutionProvider",  # fallback
    ]
    session = ort.InferenceSession(args.model, providers=providers)
    active = session.get_providers()[0]
    print(f"  Provider: {active}")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ---- Load data ----
    samples = load_test_data(args.data)
    if not samples:
        print("ERROR: no data"); sys.exit(1)

    results_dir = Path(args.output)
    combined_dir = results_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # ---- Warmup ----
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(args.warmup):
        session.run([output_name], {input_name: dummy})

    # ---- Inference ----
    all_metrics = []
    print(f"\n  {'Image':<40s} {'Dice':>6s} {'IoU':>6s} {'FPS':>7s}")
    print("  " + "-" * 60)

    for inp, mask, fname in samples:
        batch = inp[np.newaxis, ...]  # [1,3,H,W]
        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: batch})[0]
        t1 = time.perf_counter()

        lat = t1 - t0
        fps = 1.0 / lat if lat > 0 else 0

        logits = out[0, 0]
        pred_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))

        m = compute_metrics(pred_prob, mask) if mask is not None else {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "accuracy": 0}
        m["fps"] = fps
        m["latency_ms"] = lat * 1000
        m["name"] = fname
        all_metrics.append(m)

        print(f"  {fname:<40s} {m['dice']:6.3f} {m['iou']:6.3f} {fps:7.1f}")

        vis = create_combined_vis(inp, mask, pred_prob, m)
        cv2.imwrite(str(combined_dir / fname), vis)

    # ---- Summary ----
    avg = lambda k: float(np.mean([x[k] for x in all_metrics]))
    summary = {k: avg(k) for k in ["dice", "iou", "precision", "recall", "fps", "latency_ms"]}
    summary["provider"] = active
    summary["num_images"] = len(all_metrics)

    print(f"\n  Dice: {summary['dice']:.4f}  IoU: {summary['iou']:.4f}  FPS: {summary['fps']:.1f}")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "per_image": all_metrics}, f, indent=4)

    print(f"  Results saved to {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
