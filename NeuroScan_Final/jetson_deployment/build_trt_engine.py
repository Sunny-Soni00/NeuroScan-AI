import tensorrt as trt
import os

# Jetson Nano memory-safe workspace: 512 MB [cite: 524-528]
JETSON_WORKSPACE_BYTES = 512 * (1 << 20)

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Reading ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, JETSON_WORKSPACE_BYTES)

    # Enable FP16 for 2x performance boost on Jetson Nano [cite: 524-528]
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("🚀 FP16 mode enabled for maximum efficiency.")

    print(f"Building TensorRT engine... (Will take 5-10 mins on Nano)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Engine build failed.")
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ Success! Engine saved as: {engine_path}")

if __name__ == "__main__":
    build_engine("neuroscan_final.onnx", "neuroscan_final.engine")