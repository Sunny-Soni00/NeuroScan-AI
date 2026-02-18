#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine — optimised for Jetson Nano 4 GB.

Key settings for Jetson Nano:
  - Workspace limit: 512 MB  (half of 4 GB total RAM)
  - FP16 recommended (Maxwell GPU supports it, ~2× faster)
  - Fixed batch = 1

Usage:
    python convert_to_trt.py --fp16                   # recommended
    python convert_to_trt.py --fp32                   # maximum accuracy
    python convert_to_trt.py --both                   # create both engines
"""
import argparse
import sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT not installed.")
    print("On Jetson Nano it ships with JetPack.")
    sys.exit(1)

# Jetson Nano 4 GB memory-safe workspace: 512 MB
JETSON_WORKSPACE_BYTES = 512 * (1 << 20)


def convert_onnx_to_trt(onnx_path: str, trt_path: str,
                         fp16: bool = False, verbose: bool = True):
    """Convert ONNX → TensorRT engine."""
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print("\n" + "=" * 70)
    print("  ONNX → TensorRT Conversion")
    print("=" * 70)
    print(f"  Input     : {onnx_path}")
    print(f"  Output    : {trt_path}")
    print(f"  Precision : {'FP16' if fp16 else 'FP32'}")
    print(f"  Workspace : {JETSON_WORKSPACE_BYTES // (1 << 20)} MB")
    print("=" * 70 + "\n")

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print("  Parsing ONNX ...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    Error {i}: {parser.get_error(i)}")
            return False
    print("  ONNX parsed OK")

    if verbose:
        for i in range(network.num_inputs):
            t = network.get_input(i)
            print(f"    Input  [{i}] {t.name}: {t.shape}")
        for i in range(network.num_outputs):
            t = network.get_output(i)
            print(f"    Output [{i}] {t.name}: {t.shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  JETSON_WORKSPACE_BYTES)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  WARNING: FP16 not supported on this GPU, using FP32")

    print("  Building engine (may take several minutes on Nano) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  ERROR: engine build failed")
        return False

    with open(trt_path, "wb") as f:
        f.write(serialized)

    fsize = Path(trt_path).stat().st_size / (1024 * 1024)
    print(f"\n  Engine saved: {trt_path}  ({fsize:.1f} MB)")
    print("=" * 70 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT for Jetson Nano")
    parser.add_argument("--onnx", default="mobilenetv2_jetson.onnx", help="ONNX model path")
    parser.add_argument("--output", default=None, help="Custom output name")
    parser.add_argument("--fp16", action="store_true",
                        help="FP16 precision (recommended for Jetson Nano)")
    parser.add_argument("--fp32", action="store_true", help="FP32 precision")
    parser.add_argument("--both", action="store_true", help="Create both FP32 and FP16")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        print(f"ERROR: ONNX not found: {args.onnx}")
        sys.exit(1)

    stem = Path(args.onnx).stem
    jobs = []
    if args.both:
        jobs = [(False, f"{stem}_fp32.trt"), (True, f"{stem}_fp16.trt")]
    elif args.fp16:
        jobs = [(True, args.output or f"{stem}_fp16.trt")]
    else:
        jobs = [(False, args.output or f"{stem}_fp32.trt")]

    ok = 0
    for fp16, out in jobs:
        if convert_onnx_to_trt(args.onnx, out, fp16, args.verbose):
            ok += 1

    print(f"  {ok}/{len(jobs)} conversions succeeded")
    if ok:
        print("\n  Next steps:")
        print("    python run_inference_trt.py --engine <name>.trt")
    sys.exit(0 if ok == len(jobs) else 1)


if __name__ == "__main__":
    main()
