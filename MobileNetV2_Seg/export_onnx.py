#!/usr/bin/env python3
"""
Export trained MobileNetV2-UNet to ONNX  (single-file, weights inlined).
Same approach as DRUNetv2 export.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint results/mobilenetv2_best.pth.tar --output mobilenetv2_jetson.onnx
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.onnx

from model_mobilenetv2 import MobileNetV2UNet

DEFAULT_CHECKPOINT = "results/mobilenetv2_best.pth.tar"
DEFAULT_OUTPUT = "mobilenetv2_jetson.onnx"
IMG_SIZE = 256
OPSET_VERSION = 11


def export(checkpoint_path: str, output_path: str, verify: bool = True):
    device = torch.device("cpu")

    print("=" * 65)
    print("  MobileNetV2-UNet → ONNX Export")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = MobileNetV2UNet(in_channels=3, out_channels=1, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Parameters  : {n:,}")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        ref = model(dummy)
    print(f"  PyTorch out : {ref.shape}  range [{ref.min():.3f}, {ref.max():.3f}]")

    print(f"  Exporting ONNX (opset {OPSET_VERSION}) ...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Inline external weights if PyTorch split them
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        import onnx
        print("  Inlining external weights ...")
        m = onnx.load(output_path, load_external_data=True)
        onnx.save(m, output_path, save_as_external_data=False)
        if os.path.exists(data_file):
            os.remove(data_file)

    fsize = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Saved : {output_path}  ({fsize:.1f} MB)")

    # Verify
    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": dummy.numpy()})[0]
            ref_np = ref.numpy()
            max_diff = np.abs(ref_np - ort_out).max()
            sig_ref = 1.0 / (1.0 + np.exp(-np.clip(ref_np, -88, 88)))
            sig_ort = 1.0 / (1.0 + np.exp(-np.clip(ort_out, -88, 88)))
            agree = ((sig_ref > 0.5) == (sig_ort > 0.5)).mean() * 100
            print(f"\n  Verification:")
            print(f"    Max logit diff : {max_diff:.8f}")
            print(f"    Mask agreement : {agree:.4f}%")
        except ImportError:
            print("  (onnxruntime not installed — skipping verification)")

    print("\n  Next: copy ONNX to jetson_deploy/ and run on Jetson")
    print("=" * 65)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    export(args.checkpoint, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
