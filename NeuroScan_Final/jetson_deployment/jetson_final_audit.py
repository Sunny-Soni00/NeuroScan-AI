import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import glob
import time
from tqdm import tqdm

# ================= ⚙️ JETSON CONFIG =================
ENGINE_PATH = "neuroscan_final_fp16.engine" 
DATA_ROOT = "test_data"
TEST_IMAGES = sorted(glob.glob(os.path.join(DATA_ROOT, "*.png")))
TEST_MASKS = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.png")))

if len(TEST_IMAGES) != len(TEST_MASKS):
    raise SystemExit("Error: Image and Mask count mismatch!")

# ================= 🏗️ TENSORRT CLASS =================
class NeuroScanJetson:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocating buffers based on 256x256 engine input
        self.h_input = cuda.pagelocked_empty(1 * 3 * 256 * 256, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(1 * 1 * 256 * 256, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

    def infer(self, img_padded):
        np.copyto(self.h_input, img_padded.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape((256, 256))

# ================= 🚀 MAIN EXECUTION =================
if __name__ == "__main__":
    predictor = NeuroScanJetson(ENGINE_PATH)
    dice_scores, recall_scores, latencies = [], [], []

    for i in tqdm(range(len(TEST_IMAGES)), desc="Jetson Audit"):
        img = cv2.imread(TEST_IMAGES[i]).astype(np.float32)
        h, w = img.shape[0], img.shape[1]
        
        # 1. Z-Score & Padding Logic (Matches Laptop Script)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img_input = img_padded.transpose(2, 0, 1).reshape(1, 3, 256, 256)

        # 2. Inference
        start = time.time()
        pred_full = predictor.infer(img_input)
        latencies.append((time.time() - start) * 1000)

        # 3. Crop and Evaluate
        pred = pred_full[:h, :w]
        mask = cv2.imread(TEST_MASKS[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Binary Metrics
        pred_bin = (pred > 0.5).astype(np.float32)
        tp = np.sum(pred_bin * mask)
        dice = (2 * tp + 1e-7) / (np.sum(pred_bin) + np.sum(mask) + 1e-7)
        recall = (tp + 1e-7) / (np.sum(mask) + 1e-7)
        
        dice_scores.append(dice)
        recall_scores.append(recall)

    print(f"\n🏆 JETSON NANO REPORT")
    print(f"Dice: {np.mean(dice_scores)*100:.2f}% | Latency: {np.mean(latencies[10:]):.2f} ms")