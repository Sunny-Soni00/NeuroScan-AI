import torch
import time
from thop import profile
from model_3d_unet import UNet3D
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_3d():
    print(f"🚀 Starting 3D U-Net Benchmark on {torch.cuda.get_device_name(0)}...")
    model = UNet3D().to(DEVICE)
    model.eval()

    # Input: (Batch, Channels, Depth, Height, Width)
    # 128x128x128 is standard sub-volume for BraTS
    dummy_input = torch.randn(1, 1, 128, 128, 128).to(DEVICE)

    # 1. Complexity Analysis
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\n📊 Complexity:")
    print(f"🔹 3D FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"🔹 3D Params: {params / 1e6:.2f} Million")

    # 2. Latency Benchmarking (Warming up)
    for _ in range(5):
        _ = model(dummy_input)

    print("\n⏱️ Measuring Latency...")
    start_time = time.time()
    iterations = 20
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            torch.cuda.synchronize() # Wait for GPU to finish
    
    avg_latency = (time.time() - start_time) / iterations * 1000 # in ms
    print(f"🔹 Average 3D Latency: {avg_latency:.2f} ms per volume")

    # Save for the paper
    with open("3d_benchmark_results.txt", "w") as f:
        f.write(f"3D U-Net Benchmark Report\n")
        f.write(f"GFLOPs: {flops / 1e9:.2f}\n")
        f.write(f"Latency: {avg_latency:.2f} ms\n")

if __name__ == "__main__":
    benchmark_3d()