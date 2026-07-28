import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

def run_benchmark(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    
    # Setup I/O buffers (3 channels for 2.5D stacking) [cite: 134-137]
    input_shape = (1, 3, 256, 256)
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    print(f"🚀 Warm-up... Starting inference loop for Power/Latency test.")
    latencies = []
    
    try:
        # Running for 500 iterations to give you time to check tegrastats
        for i in range(500):
            start_time = time.time()
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            latencies.append((time.time() - start_time) * 1000)

        avg_latency = np.mean(latencies[10:]) # Removing warm-up spike
        print("\n" + "="*40)
        print(f"🔹 Average Latency: {avg_latency:.2f} ms")
        print(f"🔹 Throughput: {1000/avg_latency:.2f} FPS")
        print("="*40)
        print("💡 Use 'tegrastats' in another terminal to note Power (W) now.")

    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    run_benchmark("neuroscan_final.engine")