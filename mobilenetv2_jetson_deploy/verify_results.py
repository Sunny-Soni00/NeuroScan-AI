#!/usr/bin/env python3
"""
Verification script — analyse inference results and compare with
expected MobileNetV2-UNet training performance.

Works with the metrics.json produced by:
  - run_inference_trt.py
  - run_inference.py
  - mobilenet/main.py
"""
import json
import sys
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ========================  ANALYSIS  ========================
TRAINING_DICE = 0.9037   # reference from v2_metrics_log.csv
TRAINING_RECALL = 0.897


def load_metrics(results_dir="results"):
    """Load metrics.json (supports both old & new format)."""
    p = Path(results_dir) / "metrics.json"
    if not p.exists():
        # try mobilenet results path
        p = Path(results_dir) / "benchmark_results.json"
    if not p.exists():
        print(f"ERROR: no metrics file in {results_dir}/")
        sys.exit(1)

    with open(p) as f:
        data = json.load(f)

    # normalise keys — both run_inference*.py and mobilenet produce
    # {"summary": {...}, "per_image": [...]}
    if "summary" in data and "per_image" in data:
        return data["summary"], data["per_image"]
    # old benchmark format
    if "accuracy_summary" in data:
        return data["accuracy_summary"], data.get("per_image_results", [])
    return data, []


def analyse(summary, per_image):
    print("\n" + "=" * 65)
    print("  MobileNetV2-UNet Inference Results Analysis")
    print("=" * 65)

    dice = summary.get("dice", summary.get("avg_dice_score", 0))
    iou  = summary.get("iou",  summary.get("avg_iou", 0))
    prec = summary.get("precision", summary.get("avg_precision", 0))
    rec  = summary.get("recall", summary.get("avg_recall", 0))
    fps  = summary.get("fps", summary.get("fps_avg", 0))

    print(f"\n  Dice:      {dice:.4f}   (training ref: {TRAINING_DICE:.4f})")
    print(f"  IoU:       {iou:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}   (training ref: {TRAINING_RECALL:.4f})")
    print(f"  FPS:       {fps:.1f}")

    # gap analysis
    gap = TRAINING_DICE - dice
    if gap < 0.02:
        tag = "EXCELLENT — matches training"
    elif gap < 0.05:
        tag = "GOOD — small gap, acceptable"
    elif gap < 0.10:
        tag = "MODERATE — check preprocessing matches training"
    else:
        tag = "POOR — likely preprocessing mismatch (2.5D?)"
    print(f"\n  Dice gap from training: {gap:+.4f}  =>  {tag}")

    if fps >= 15:
        print("  FPS: suitable for real-time on Jetson Nano")
    elif fps >= 8:
        print("  FPS: acceptable; consider FP16 for more headroom")
    elif fps > 0:
        print("  FPS: low — try FP16 engine or reduce resolution")

    # per-image
    if per_image:
        dices = [
            img.get("dice", img.get("dice_score", 0))
            for img in per_image
        ]
        best_idx = int(np.argmax(dices))
        worst_idx = int(np.argmin(dices))
        print(f"\n  Per-image ({len(dices)} samples):")
        print(f"    Best:  {dices[best_idx]:.4f}  ({per_image[best_idx].get('name', best_idx)})")
        print(f"    Worst: {dices[worst_idx]:.4f}  ({per_image[worst_idx].get('name', worst_idx)})")
        print(f"    Std:   {np.std(dices):.4f}")

        failed = [
            img for img, d in zip(per_image, dices) if d < 0.5
        ]
        if failed:
            print(f"\n  {len(failed)} images with Dice < 0.5:")
            for img in failed[:5]:
                n = img.get("name", "?")
                d = img.get("dice", img.get("dice_score", 0))
                print(f"    - {n}: Dice={d:.3f}")

        # distribution plot
        if HAS_PLT:
            _plot_distribution(dices)

    # combined vis check
    combined = Path("results/combined")
    if combined.is_dir():
        n = len(list(combined.glob("*.png")))
        print(f"\n  Visualisations: {n} combined images in results/combined/")
        print("    Legend: green=TP, red=FP, blue=FN")

    print("\n" + "=" * 65 + "\n")


def _plot_distribution(dices):
    plt.figure(figsize=(9, 5))
    plt.hist(dices, bins=20, color="steelblue", edgecolor="black", alpha=0.75)
    plt.axvline(np.mean(dices), color="red", ls="--", lw=2,
                label=f"Mean {np.mean(dices):.3f}")
    plt.axvline(TRAINING_DICE, color="green", ls=":", lw=2,
                label=f"Training ref {TRAINING_DICE:.3f}")
    plt.xlabel("Dice Score")
    plt.ylabel("Count")
    plt.title("Dice Score Distribution — TRT / ONNX Inference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = Path("results/dice_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Distribution plot saved to {out}")


def main():
    summary, per_image = load_metrics()
    analyse(summary, per_image)


if __name__ == "__main__":
    main()
