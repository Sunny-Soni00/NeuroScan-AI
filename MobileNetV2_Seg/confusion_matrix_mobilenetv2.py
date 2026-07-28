#!/usr/bin/env python3
"""
Compute pixel-wise confusion matrix for MobileNetV2-UNet ONNX predictions.

Creates and saves:
1) Count confusion matrix plot
2) Percentage confusion matrix plot
3) Raw matrices in JSON

Usage:
    python confusion_matrix_mobilenetv2.py
    python confusion_matrix_mobilenetv2.py --show
    python confusion_matrix_mobilenetv2.py \
        --model mobilenetv2_jetson.onnx \
        --data ../mobilenetv2_jetson_deploy/test_data \
        --output results/laptop_test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


IMG_SIZE = 256
THRESHOLD = 0.5


def load_test_data(data_dir: str):
    """
    Load test samples.
    Prefers npz/ (true 2.5D stacks), falls back to PNG + masks/.
    Returns list of tuples: (input_stack[3,H,W], gt_mask[H,W], filename)
    """
    npz_dir = os.path.join(data_dir, "npz")
    samples = []

    if os.path.isdir(npz_dir):
        for p in sorted(Path(npz_dir).glob("*.npz")):
            d = np.load(str(p), allow_pickle=True)
            if "input" not in d or "mask" not in d:
                continue
            fname = str(d["filename"]) if "filename" in d else p.with_suffix(".png").name
            samples.append((
                d["input"].astype(np.float32),
                d["mask"].astype(np.float32),
                fname,
            ))
        if samples:
            return samples

    mask_dir = os.path.join(data_dir, "masks")
    for p in sorted(Path(data_dir).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        inp = np.stack([img, img, img], axis=0)

        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mpath = os.path.join(mask_dir, p.name)
        if os.path.exists(mpath):
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)

        samples.append((inp, mask, p.name))

    return samples


def run_onnx_inference(model_path: str, samples):
    """Run ONNX model and return list of (pred_binary, gt_binary, filename)."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out = []
    for inp, gt, fname in samples:
        batch = inp[np.newaxis, ...].astype(np.float32)
        logits = session.run([output_name], {input_name: batch})[0][0, 0]
        pred_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))

        pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)
        out.append((pred_bin, gt_bin, fname))

    return out


def compute_confusion(pred_gt_list):
    """Return confusion matrix [[TN, FP], [FN, TP]] as int64."""
    tn = fp = fn = tp = 0

    for pred, gt, _ in pred_gt_list:
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        tp += int(np.sum((pred == 1) & (gt == 1)))
        tn += int(np.sum((pred == 0) & (gt == 0)))
        fp += int(np.sum((pred == 1) & (gt == 0)))
        fn += int(np.sum((pred == 0) & (gt == 1)))

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def save_confusion_plots(cm: np.ndarray, out_dir: Path, show: bool):
    """Save count and percent confusion matrix plots."""
    labels = ["Background (0)", "Tumor (1)"]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Pixel-wise Confusion Matrix (MobileNetV2-UNet)")
    plt.tight_layout()
    count_plot_path = out_dir / "pixel_confusion_matrix_counts.png"
    plt.savefig(count_plot_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    total = cm.sum()
    cm_percent = (cm / (total + 1e-12)) * 100.0

    annot_percent = np.array([[f"{v:.2f}%" for v in row] for row in cm_percent])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_percent, annot=annot_percent, fmt="", cmap="Greens", cbar=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Pixel-wise Confusion Matrix (%) (MobileNetV2-UNet)")
    plt.tight_layout()
    percent_plot_path = out_dir / "pixel_confusion_matrix_percent.png"
    plt.savefig(percent_plot_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    return cm_percent, count_plot_path, percent_plot_path


def main():
    parser = argparse.ArgumentParser(description="Compute pixel-wise confusion matrix for MobileNetV2-UNet")
    parser.add_argument("--model", default="mobilenetv2_jetson.onnx", help="Path to ONNX model")
    parser.add_argument("--data", default="../mobilenetv2_jetson_deploy/test_data", help="Path to test_data folder")
    parser.add_argument("--output", default="results/laptop_test", help="Output folder for plots and JSON")
    parser.add_argument("--show", action="store_true", help="Display plots with plt.show()")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    if not os.path.isdir(args.data):
        print(f"ERROR: data folder not found: {args.data}")
        sys.exit(1)

    samples = load_test_data(args.data)
    if not samples:
        print("ERROR: no test samples found")
        sys.exit(1)

    pred_gt_list = run_onnx_inference(args.model, samples)
    cm = compute_confusion(pred_gt_list)
    cm_percent, count_plot_path, percent_plot_path = save_confusion_plots(cm, out_dir, show=args.show)

    out_json = {
        "num_images": len(samples),
        "threshold": THRESHOLD,
        "confusion_matrix_counts": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "matrix": cm.tolist(),
            "layout": "[[TN, FP], [FN, TP]]",
        },
        "confusion_matrix_percent": {
            "matrix": cm_percent.tolist(),
            "unit": "percent_of_all_pixels",
        },
        "saved_plots": {
            "counts": str(count_plot_path),
            "percent": str(percent_plot_path),
        },
    }

    json_path = out_dir / "pixel_confusion_matrix.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=4)

    print("=" * 70)
    print("Pixel-wise Confusion Matrix (MobileNetV2-UNet)")
    print("=" * 70)
    print(f"Images: {len(samples)}")
    print("Counts matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nPercent matrix (% of all pixels):")
    print(np.round(cm_percent, 4))
    print(f"\nSaved: {count_plot_path}")
    print(f"Saved: {percent_plot_path}")
    print(f"Saved: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()