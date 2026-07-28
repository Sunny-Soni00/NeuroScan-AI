import torch
import time
from model_drunet_v2 import AttentionDRUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_25d():
    print(f"🚀 Benchmarking 2.5D DRUNet v2 on {torch.cuda.get_device_name(0)}...")
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    model.eval()

    # Input: (Batch, Channels, Height, Width)
    # Our 2.5D input is 3 channels (prev, current, next)
    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    print("⏱️ Measuring Latency...")
    start_time = time.time()
    iterations = 100 # Zyada iterations for accuracy
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            torch.cuda.synchronize() 
    
    avg_latency = (time.time() - start_time) / iterations * 1000 # in ms
    print(f"🔹 Average 2.5D Latency: {avg_latency:.2f} ms per slice")

    # Record this for your report
    with open("PROOFS/25d_real_latency.txt", "w") as f:
        f.write(f"2.5D Real Latency: {avg_latency:.2f} ms\n")

if __name__ == "__main__":
    benchmark_25d()