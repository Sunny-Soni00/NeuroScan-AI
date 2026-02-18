#!/usr/bin/env python3
"""
MobileNetV2-UNet TensorRT Inference for Jetson Nano 4 GB
========================================================
Loads 2.5D test stacks (exported by export_test_data.py) and runs TensorRT
inference on the GPU.  Produces per-image metrics and a combined
Input | Ground Truth | Prediction | Overlap visualisation.

Usage (on Jetson Nano):
    # FP16 engine (recommended for Nano):
    python run_inference_trt.py --engine mobilenetv2_jetson_fp16.trt

    # FP32 engine:
    python run_inference_trt.py --engine mobilenetv2_jetson_fp32.trt
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Attempt to import tegrastats / jtop for Jetson monitoring
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT is not installed.  On Jetson Nano it ships with JetPack.")
    sys.exit(1)

# ========================  CONFIG  ========================
IMG_SIZE = 256
THRESHOLD = 0.5


# ========================  TRT ENGINE  ========================
class TRTInference:
    """
    TensorRT inference engine using raw CUDA calls (no PyCUDA dependency).
    Optimised for Jetson Nano 4 GB:
      - Single-stream async inference
      - Pre-allocated GPU buffers (no per-frame malloc)
    """

    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Tensor names & shapes
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        print(f"  TRT Engine loaded: {engine_path}")
        print(f"    Input  : {self.input_name}  {self.input_shape}")
        print(f"    Output : {self.output_name}  {self.output_shape}")

        # Load CUDA runtime
        try:
            self.cuda = ctypes.CDLL("libcudart.so")
        except OSError:
            self.cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

        # Pre-allocate GPU memory (once, reused every frame)
        self.input_nbytes = int(np.prod(self.input_shape)) * 4   # float32
        self.output_nbytes = int(np.prod(self.output_shape)) * 4

        self.d_input = ctypes.c_void_p()
        self.d_output = ctypes.c_void_p()
        assert self.cuda.cudaMalloc(ctypes.byref(self.d_input),
                                     ctypes.c_size_t(self.input_nbytes)) == 0
        assert self.cuda.cudaMalloc(ctypes.byref(self.d_output),
                                     ctypes.c_size_t(self.output_nbytes)) == 0

        self.stream = ctypes.c_void_p()
        assert self.cuda.cudaStreamCreate(ctypes.byref(self.stream)) == 0

    # ----------------------------------------------------------
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on a PREPROCESSED input [1, 3, 256, 256] float32.
        Returns raw logits [1, 1, 256, 256].
        """
        assert input_tensor.shape == self.input_shape, \
            f"Shape mismatch: expected {self.input_shape}, got {input_tensor.shape}"
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        output = np.empty(self.output_shape, dtype=np.float32)

        # H2D
        self.cuda.cudaMemcpyAsync(
            self.d_input,
            input_tensor.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.input_nbytes), 1, self.stream)

        # Set addresses & execute
        self.context.set_tensor_address(self.input_name, int(self.d_input.value))
        self.context.set_tensor_address(self.output_name, int(self.d_output.value))
        self.context.execute_async_v3(int(self.stream.value))

        # D2H
        self.cuda.cudaMemcpyAsync(
            output.ctypes.data_as(ctypes.c_void_p),
            self.d_output,
            ctypes.c_size_t(self.output_nbytes), 2, self.stream)

        self.cuda.cudaStreamSynchronize(self.stream)
        return output

    # ----------------------------------------------------------
    def __del__(self):
        try:
            self.cuda.cudaFree(self.d_input)
            self.cuda.cudaFree(self.d_output)
            self.cuda.cudaStreamDestroy(self.stream)
        except Exception:
            pass


# ========================  DATA LOADING  ========================
def load_test_data_npz(data_dir: str):
    """
    Load 2.5D test stacks from .npz files produced by export_test_data.py.
    Each npz contains: input[3,256,256], mask[256,256], filename.
    """
    npz_dir = os.path.join(data_dir, "npz")
    samples = []

    if os.path.isdir(npz_dir):
        for p in sorted(Path(npz_dir).glob("*.npz")):
            d = np.load(str(p), allow_pickle=True)
            inp = d["input"]                         # [3, 256, 256]  float32
            mask = d["mask"]                         # [256, 256]     float32
            fname = str(d["filename"])
            samples.append((inp, mask, fname))
        if samples:
            print(f"  Loaded {len(samples)} 2.5D stacks from {npz_dir}/")
            return samples

    # ---- Fallback: load PNGs (replicate same slice 3×) ----
    print("  WARNING: npz/ not found, falling back to PNG loading (no real 2.5D context)")
    mask_dir = os.path.join(data_dir, "masks")
    for p in sorted(Path(data_dir).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        inp = np.stack([img, img, img], axis=0)      # replicate (lossy)

        mask_path = os.path.join(mask_dir, p.name)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)

        samples.append((inp, mask, p.name))

    print(f"  Loaded {len(samples)} images (PNG fallback)")
    return samples


# ========================  METRICS  ========================
def calculate_metrics(pred_prob: np.ndarray, gt: np.ndarray):
    """Dice, IoU, Precision, Recall from probability map and binary GT."""
    pred = (pred_prob > THRESHOLD).astype(np.float32)
    gt = gt.astype(np.float32)

    tp = np.sum(pred * gt)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return float(dice), float(iou), float(precision), float(recall)


# ========================  VISUALISATION  ========================
def create_combined_visualisation(input_stack, gt_mask, pred_prob, metrics_dict):
    """
    Build a 4-panel image:  Input | Ground Truth | Prediction | Overlap
    Using the current (middle) slice of the 2.5D stack as the input view.

    Overlap colour scheme:
      - Green  = True Positive  (both GT and Pred)
      - Red    = False Positive (Pred only)
      - Blue   = False Negative (GT only)
      - Base   = greyscale MRI
    """
    H, W = IMG_SIZE, IMG_SIZE
    current_slice = input_stack[1]  # middle channel = current slice

    # --- Panel 1: Input (greyscale → BGR) ---
    input_u8 = (current_slice * 255).clip(0, 255).astype(np.uint8)
    input_bgr = cv2.cvtColor(input_u8, cv2.COLOR_GRAY2BGR)

    # --- Panel 2: Ground Truth (green mask on dark background) ---
    gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    gt_vis[:, :, 1] = gt_binary * 255  # green channel

    # --- Panel 3: Prediction (red mask on dark background) ---
    pred_binary = (pred_prob > THRESHOLD).astype(np.uint8)
    pred_vis = np.zeros((H, W, 3), dtype=np.uint8)
    pred_vis[:, :, 2] = pred_binary * 255  # red channel (BGR)

    # --- Panel 4: Overlap on MRI ---
    overlap = input_bgr.copy()
    tp_mask = (pred_binary & gt_binary).astype(bool)
    fp_mask = (pred_binary & ~gt_binary).astype(bool)
    fn_mask = (~pred_binary & gt_binary).astype(bool)
    # Green = TP, Red = FP, Blue = FN
    overlap[tp_mask] = [0, 255, 0]
    overlap[fp_mask] = [0, 0, 255]
    overlap[fn_mask] = [255, 0, 0]

    # Blend overlay with original for better visibility
    alpha = 0.55
    blended = input_bgr.copy()
    blended[tp_mask] = (alpha * np.array([0, 255, 0]) + (1 - alpha) * blended[tp_mask]).astype(np.uint8)
    blended[fp_mask] = (alpha * np.array([0, 0, 255]) + (1 - alpha) * blended[fp_mask]).astype(np.uint8)
    blended[fn_mask] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * blended[fn_mask]).astype(np.uint8)

    # Combine 4 panels side-by-side
    combined = np.hstack([input_bgr, gt_vis, pred_vis, blended])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Input",   (10,       25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "GT",      (W + 10,   25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "Pred",    (2*W + 10, 25), font, 0.6, (0, 0, 255), 2)
    cv2.putText(combined, "Overlap", (3*W + 10, 25), font, 0.6, (255, 255, 255), 2)

    # Metrics bar at bottom
    if metrics_dict:
        bar = np.zeros((35, 4 * W, 3), dtype=np.uint8)
        txt = (f"Dice: {metrics_dict['dice']:.3f}  |  "
               f"IoU: {metrics_dict['iou']:.3f}  |  "
               f"Prec: {metrics_dict['precision']:.3f}  |  "
               f"Recall: {metrics_dict['recall']:.3f}  |  "
               f"FPS: {metrics_dict.get('fps', 0):.1f}")
        cv2.putText(bar, txt, (10, 24), font, 0.55, (220, 220, 220), 1)
        combined = np.vstack([combined, bar])

    # Legend bar
    legend = np.zeros((25, 4 * W, 3), dtype=np.uint8)
    cv2.putText(legend, "Green=TP   Red=FP   Blue=FN", (10, 18),
                font, 0.5, (180, 180, 180), 1)
    combined = np.vstack([combined, legend])

    return combined


# ========================  SYSTEM MONITORING  ========================
def get_system_stats():
    """Collect CPU/RAM and (if available) GPU stats on Jetson."""
    import psutil
    proc = psutil.Process()
    stats = {
        "cpu_percent": proc.cpu_percent(),
        "ram_mb": proc.memory_info().rss / (1024 * 1024),
    }
    # Check for Jetson GPU info via tegrastats or jtop
    if JTOP_AVAILABLE:
        try:
            with jtop() as j:
                if j.ok():
                    stats["gpu_percent"] = j.stats.get("GPU", 0)
                    stats["gpu_temp_c"] = j.stats.get("Temp GPU", 0)
        except Exception:
            pass
    return stats


# ========================  MAIN  ========================
def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 TRT inference on Jetson Nano")
    parser.add_argument("--engine", type=str, default="mobilenetv2_jetson_fp16.trt",
                        help="TRT engine file (.trt)")
    parser.add_argument("--data", type=str, default="test_data",
                        help="Test data directory (with npz/ subfolder)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  MobileNetV2-UNet TensorRT Inference  —  Jetson Nano 4 GB")
    print("=" * 70)

    # --- Initialise ---
    model = TRTInference(args.engine)
    samples = load_test_data_npz(args.data)
    if not samples:
        print("ERROR: no test data found"); sys.exit(1)

    # Output dirs
    results_dir = Path(args.output)
    combined_dir = results_dir / "combined"
    pred_dir = results_dir / "pred"
    gt_dir = results_dir / "gt"
    overlay_dir = results_dir / "overlay"
    for d in [results_dir, combined_dir, pred_dir, gt_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Warmup (GPU needs a few passes before stable timing) ---
    print(f"\n  Warmup ({args.warmup} passes) ...")
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(args.warmup):
        model.infer(dummy)

    # --- Inference loop ---
    all_metrics = []
    total_inference_time = 0.0

    print(f"\n  Running inference on {len(samples)} images ...\n")
    print(f"  {'Image':<40s} {'Dice':>6s} {'IoU':>6s} {'Prec':>6s} {'Recall':>7s} {'FPS':>7s}")
    print("  " + "-" * 74)

    for inp_stack, gt_mask, fname in samples:
        # Add batch dimension: [3,256,256] → [1,3,256,256]
        input_batch = inp_stack[np.newaxis, ...].astype(np.float32)

        # Timed inference
        t0 = time.perf_counter()
        raw_output = model.infer(input_batch)
        t1 = time.perf_counter()

        latency = t1 - t0
        total_inference_time += latency
        fps = 1.0 / latency if latency > 0 else 0

        # Post-process: sigmoid on logits
        logits = raw_output[0, 0]  # [H, W]
        pred_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))

        # Metrics
        dice, iou, precision, recall = calculate_metrics(pred_prob, gt_mask)
        m = {"name": fname, "dice": dice, "iou": iou, "precision": precision,
             "recall": recall, "fps": fps, "latency_ms": latency * 1000}
        all_metrics.append(m)

        print(f"  {fname:<40s} {dice:6.3f} {iou:6.3f} {precision:6.3f} {recall:7.3f} {fps:7.1f}")

        # ----- Save visualisations -----
        # Combined: Input | GT | Pred | Overlap
        combined_img = create_combined_visualisation(inp_stack, gt_mask, pred_prob, m)
        cv2.imwrite(str(combined_dir / fname), combined_img)

        # Individual panels
        pred_u8 = ((pred_prob > THRESHOLD).astype(np.uint8) * 255)
        gt_u8 = ((gt_mask > 0.5).astype(np.uint8) * 255)
        cv2.imwrite(str(pred_dir / fname), pred_u8)
        cv2.imwrite(str(gt_dir / fname), gt_u8)

        # Overlay only (blended on MRI)
        current_u8 = (inp_stack[1] * 255).clip(0, 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(current_u8, cv2.COLOR_GRAY2BGR)
        pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)
        gt_bin = (gt_mask > 0.5).astype(np.uint8)
        overlay_bgr[pred_bin & gt_bin > 0] = [0, 255, 0]       # TP green
        overlay_bgr[(pred_bin > 0) & (gt_bin == 0)] = [0, 0, 255]  # FP red
        overlay_bgr[(pred_bin == 0) & (gt_bin > 0)] = [255, 0, 0]  # FN blue
        cv2.imwrite(str(overlay_dir / fname), overlay_bgr)

    # --- Summary ---
    N = len(all_metrics)
    avg_dice = np.mean([m["dice"] for m in all_metrics])
    avg_iou = np.mean([m["iou"] for m in all_metrics])
    avg_prec = np.mean([m["precision"] for m in all_metrics])
    avg_rec = np.mean([m["recall"] for m in all_metrics])
    avg_fps = np.mean([m["fps"] for m in all_metrics])
    avg_lat = np.mean([m["latency_ms"] for m in all_metrics])

    summary = {
        "num_images": N,
        "engine": args.engine,
        "dice": float(avg_dice),
        "iou": float(avg_iou),
        "precision": float(avg_prec),
        "recall": float(avg_rec),
        "fps_avg": float(avg_fps),
        "latency_ms_avg": float(avg_lat),
        "latency_ms_min": float(np.min([m["latency_ms"] for m in all_metrics])),
        "latency_ms_max": float(np.max([m["latency_ms"] for m in all_metrics])),
        "total_inference_sec": float(total_inference_time),
    }

    print()
    print("  " + "=" * 50)
    print(f"  Segmentation  — Dice: {avg_dice:.4f}  IoU: {avg_iou:.4f}")
    print(f"  Precision: {avg_prec:.4f}   Recall: {avg_rec:.4f}")
    print(f"  Performance   — FPS: {avg_fps:.1f}  Latency: {avg_lat:.1f} ms")
    print("  " + "=" * 50)

    # Try to get system stats
    try:
        sys_stats = get_system_stats()
        summary["system"] = sys_stats
        print(f"  RAM: {sys_stats.get('ram_mb', 0):.0f} MB")
        if "gpu_percent" in sys_stats:
            print(f"  GPU: {sys_stats['gpu_percent']}%")
    except Exception:
        pass

    # Save JSON
    json_path = results_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_image": all_metrics}, f, indent=4)

    print(f"\n  Results saved to {results_dir}/")
    print(f"    combined/  : Input | GT | Pred | Overlap  (per image)")
    print(f"    pred/      : Binary predictions")
    print(f"    gt/        : Ground truth masks")
    print(f"    overlay/   : Blended overlays on MRI")
    print(f"    metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()