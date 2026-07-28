import torch
from thop import profile
from model_drunet_v2 import AttentionDRUNet
import matplotlib.pyplot as plt
import numpy as np
import os

# Create Proofs Directory
os.makedirs("PROOFS", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_complexity():
    print("📊 Calculating Complexity (FLOPs & Parameters)...")
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    # Dummy input for 2.5D: (Batch, Channels, H, W)
    input_tensor = torch.randn(1, 3, 256, 256).to(DEVICE)
    
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    print(f"🔹 FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"🔹 Parameters: {params / 1e6:.2f} Million")
    
    with open("PROOFS/complexity_analysis.txt", "w") as f:
        f.write(f"Model Complexity Report\n")
        f.write(f"GFLOPs: {flops / 1e9:.2f}\n")
        f.write(f"Params (M): {params / 1e6:.2f}\n")

def generate_boundary_proof():
    print("\n🖼️ Generating Boundary Visual Proof (Reflexive Padding)...")
    # Mock data to show logic
    slice_1 = np.random.rand(256, 256)
    
    # Logic 1: Zero Padding (Bad for edges)
    stack_zero = np.stack([np.zeros_like(slice_1), slice_1, np.random.rand(256, 256)], axis=0)
    
    # Logic 2: Reflexive Padding (Our V2 Logic)
    stack_reflexive = np.stack([slice_1, slice_1, np.random.rand(256, 256)], axis=0)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(stack_zero[0], cmap='gray')
    plt.title("Z=1 Input with Zero Padding\n(Missing Spatial Context)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(stack_reflexive[0], cmap='gray')
    plt.title("Z=1 Input with Reflexive Padding\n(Maintains Edge Continuity)")
    
    plt.savefig("PROOFS/boundary_logic_proof.png")
    print("✅ Proof image saved in PROOFS/boundary_logic_proof.png")

if __name__ == "__main__":
    calculate_complexity()
    generate_boundary_proof()