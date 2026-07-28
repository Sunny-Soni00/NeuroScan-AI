import torch
import sys
import os
from thop import profile, clever_format

# Add model folders to Python path
sys.path.append(os.path.join(os.getcwd(), "DRUnet_v2"))
sys.path.append(os.path.join(os.getcwd(), "MobileNetV2_Seg"))

# Import DRUnet V2 model
try:
    from model_drunet_v2 import AttentionDRUNet
    print("✅ DRUnet_v2 model found.")
except ImportError as e:
    print(f"❌ Error importing DRUnet: {e}")

# Import MobileNetV2 model
try:
    from model_mobilenetv2 import MobileNetV2UNet 
    print("✅ MobileNetV2 model found.")
except ImportError as e:
    print(f"❌ Error importing MobileNetV2: {e}")

def run_complexity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create dummy input: batch=1, channels=3, height=256, width=256
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # Profile DRUnet V2: measure FLOPs and parameters
    print("\n" + "="*30 + "\nPROFILING DRUnet_v2\n" + "="*30)
    model_v2 = AttentionDRUNet(in_channels=3, out_channels=1).to(device)
    flops_v2, params_v2 = profile(model_v2, inputs=(input_tensor, ), verbose=False)
    f_v2, p_v2 = clever_format([flops_v2, params_v2], "%.3f")
    print(f"Total FLOPs: {f_v2}")
    print(f"Total Params: {p_v2}")

    # Profile MobileNetV2: measure FLOPs and parameters
    print("\n" + "="*30 + "\nPROFILING MobileNetV2-UNet\n" + "="*30)
    model_mb = MobileNetV2UNet(in_channels=3, out_channels=1).to(device)
    flops_mb, params_mb = profile(model_mb, inputs=(input_tensor, ), verbose=False)
    f_mb, p_mb = clever_format([flops_mb, params_mb], "%.3f")
    print(f"Total FLOPs: {f_mb}")
    print(f"Total Params: {p_mb}")

if __name__ == "__main__":
    run_complexity_check()