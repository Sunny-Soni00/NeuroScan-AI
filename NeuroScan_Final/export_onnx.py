import torch
from model_drunet_v2 import AttentionDRUNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model_v2.pth"
ONNX_OUTPUT = "neuroscan_final.onnx"

def export_to_onnx():
    # Load model architecture
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Safely load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Tracing dummy input (3 channels for 2.5D logic) [cite: 134-137]
    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)

    print(f"Exporting model to {ONNX_OUTPUT}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_OUTPUT, 
        export_params=True, 
        opset_version=18,  # Increased to 18 to fix the error you got
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output']
    )
    print("✅ ONNX Export Successful. Now transfer this file to Jetson Nano.")

if __name__ == "__main__":
    export_to_onnx()