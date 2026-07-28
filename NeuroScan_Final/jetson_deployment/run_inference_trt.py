#!/usr/bin/env python3
"""
NeuroScan AI: Jetson Nano TensorRT Inference & Power Audit
=========================================================
Integrated Metrics: Dice, Recall, Latency, Power (W), Accuracy per Watt.
Visuals: Input | Ground Truth | Prediction | Overlap
"""

import os
import sys
import time
import ctypes
import json
import numpy as np
import cv2
import argparse
from pathlib import Path

# Try to import jtop for power monitoring
try:
    from jtop import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False
    print("⚠️ Warning: 'jtop' not installed. Accuracy per Watt will not be calculated.")

try:
    import tensorrt as trt
except ImportError:
    print("❌ ERROR: TensorRT not found. Ensure you are running this on the Jetson Nano.")
    sys.exit(1)

# --- Configuration ---
IMG_SIZE = 256
THRESHOLD = 0.5

class NeuroScanInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # Load CUDA Runtime via ctypes to minimize dependencies
        try:
            self.cuda = ctypes.CDLL("libcudart.so")
        except OSError:
            self.cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

        # Memory Allocation (float32 = 4 bytes)
        self.input_nbytes = int(np.prod(self.input_shape)) * 4
        self.output_nbytes = int(np.prod(self.output_shape)) * 4

        self.d_input = ctypes.c_void_p()
        self.d_output = ctypes.c_void_p()
        self.cuda.cudaMalloc(ctypes.byref(self.d_input), ctypes.c_size_t(self.input_nbytes))
        self.cuda.cudaMalloc(ctypes.byref(self.d_output), ctypes.c_size_t(self.output_nbytes))

        self.stream = ctypes.c_void_p()
        self.cuda.cudaStreamCreate(ctypes.byref(self.stream))

    def infer(self, input_tensor):
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        output = np.empty(self.output_shape, dtype=np.float32)

        # H2D Transfer
        self.cuda.cudaMemcpyAsync(self.d_input, input_tensor.ctypes.data_as(ctypes.c_void_p), 
                                 ctypes.c_size_t(self.input_nbytes), 1, self.stream)

        # Execute (V3 API for compatibility)
        self.context.set_tensor_address(self.input_name, int(self.d_input.value))
        self.context.set_tensor_address(self.output_name, int(self.d_output.value))
        self.context.execute_async_v3(int(self.stream.value))

        # D2H Transfer
        self.cuda.cudaMemcpyAsync(output.ctypes.data_as(ctypes.c_void_p), self.d_output, 
                                 ctypes.c_size_t(self.output_nbytes), 2, self.stream)
        
        self.cuda.cudaStreamSynchronize(self.stream)
        # Apply Sigmoid
        return 1.0 / (1.0 + np.exp(-np.clip(output[0, 0], -88, 88)))

def calculate_metrics(pred_prob, gt):
    pred = (pred_prob > THRESHOLD).astype(np.float32)
    tp = np.sum(pred * gt)
    dice = (2 * tp + 1e-7) / (np.sum(pred) + np.sum(gt) + 1e-7)
    recall = (tp + 1e-7) / (np.sum(gt) + 1e-7)
    return dice, recall

def create_vis(mri, gt, pred, dice, recall):
    """Generates Side-by-Side: Input | GT | Pred | Overlap"""
    mri_u8 = (mri * 255).astype(np.uint8)
    mri_bgr = cv2.cvtColor(mri_u8, cv2.COLOR_GRAY2BGR)
    
    gt_viz = np.zeros((256, 256, 3), dtype=np.uint8)
    gt_viz[:, :, 1] = (gt * 255).astype(np.uint8) # Green
    
    pred_viz = np.zeros((256, 256, 3), dtype=np.uint8)
    pred_viz[:, :, 2] = ((pred > THRESHOLD) * 255).astype(np.uint8) # Red
    
    overlap = mri_bgr.copy()
    mask_tp = ((pred > THRESHOLD) & (gt > 0.5))
    overlap[mask_tp] = [0, 255, 0] # TP in Green
    
    res = np.hstack([mri_bgr, gt_viz, pred_viz, overlap])
    cv2.putText(res, f"Dice: {dice:.2f} Recall: {recall:.2f}", (10, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="neuroscan_final_fp16.engine")
    parser.add_argument("--data", default="test_data")
    args = parser.parse_args()

    model = NeuroScanInference(args.engine)
    img_paths = sorted(glob.glob(os.path.join(args.data, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(args.data, "masks", "*.png")))
    
    results = []
    power_samples = []
    
    print(f"🚀 Starting Audit on Jetson Nano...")
    
    # Using jtop as a context manager to track power draw
    with jtop() if HAS_JTOP else open(os.devnull, 'w') as jetson:
        for i in range(len(img_paths)):
            img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            # 2.5D Stack (Simplification for audit)
            inp_stack = np.stack([img, img, img], axis=0).reshape(1, 3, 256, 256)
            gt = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            t0 = time.perf_counter()
            pred = model.infer(inp_stack)
            latency = (time.perf_counter() - t0) * 1000
            
            # Sample Power Draw (Total system power in Watts)
            if HAS_JTOP and jetson.ok():
                power_samples.append(jetson.power['tot']['avg'] / 1000.0)

            dice, recall = calculate_metrics(pred, gt)
            results.append({'dice': dice, 'recall': recall, 'latency': latency})

            # Save Visual Comparison
            vis = create_vis(img, gt, pred, dice, recall)
            cv2.imwrite(f"results/staged_{i}.png", vis)

    # --- FINAL REPORT ---
    avg_dice = np.mean([r['dice'] for r in results])
    avg_lat = np.mean([r['latency'] for r in results])
    avg_power = np.mean(power_samples) if power_samples else 0
    apw = (avg_dice * 100) / avg_power if avg_power > 0 else 0

    print("\n" + "="*50)
    print(f"🏆 NEUROSCAN JETSON AUDIT COMPLETE")
    print("="*50)
    print(f"🔹 Avg Dice Score    : {avg_dice*100:.2f}%")
    print(f"🔹 Avg Latency       : {avg_lat:.2f} ms")
    print(f"🔹 Avg Power Draw    : {avg_power:.2f} W")
    print(f"🔹 Accuracy per Watt : {apw:.4f} (%/W)")
    print("="*50)

if __name__ == "__main__":
    import glob
    os.makedirs("results", exist_ok=True)
    main()