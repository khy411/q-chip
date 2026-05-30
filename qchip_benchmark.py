import onnxruntime as ort
import numpy as np
import cv2, os, time, psutil, glob

# Config
FP32_MODEL = "qchip_fp32.onnx"
INT8_MODEL = "qchip_int8.onnx"
VAL_IMAGES = "val_images"
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20
IMG_SIZE = 640

CLASSES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :]
    return img

def benchmark_model(model_path, images):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(WARMUP_RUNS):
        inp = preprocess(images[i % len(images)])
        sess.run(None, {input_name: inp})

    # Benchmark latency
    latencies = []
    process = psutil.Process(os.getpid())

    ram_before = process.memory_info().rss / 1e6 # MB

    for i in range(BENCHMARK_RUNS):
        inp = preprocess(images[i % len(images)])
        start = time.perf_counter()
        sess.run(None, {input_name: inp})
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms

    ram_after = process.memory_info().rss / 1e6 # MB

    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    ram_used = ram_after - ram_before
    model_mb = os.path.getsize(model_path) / 1e6

    return {
        "model": os.path.basename(model_path),
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "ram_mb": ram_used,
        "size_mb": model_mb,
    }

def print_results(r):
    print(f"\n{'='*45}")
    print(f"Model: {r['model']}")
    print(f"Size: {r['size_mb']:.1f} MB")
    print(f"Avg: {r['avg_ms']:.1f} ms/image")
    print(f"Std: {r['std_ms']:.1f} ms")
    print(f"Min: {r['min_ms']:.1f} ms")
    print(f"Max: {r['max_ms']:.1f} ms")
    print(f"RAM: {r['ram_mb']:.1f} MB used during inference")
    print(f"{'='*45}")

if __name__ == "__main__":
    images = glob.glob(os.path.join(VAL_IMAGES, "*.jpg"))
    if not images:
        print("No images found in val_images/. Check the folder.")
        exit()

    print(f"Found {len(images)} val images. Running benchmark...")

    fp32 = benchmark_model(FP32_MODEL, images)
    int8 = benchmark_model(INT8_MODEL, images)

    print_results(fp32)
    print_results(int8)

    # Summary comparison
    speedup = fp32['avg_ms'] / int8['avg_ms']
    size_reduction = (1 - int8['size_mb'] / fp32['size_mb']) * 100

    print(f"\nQ-CHIP SUMMARY")
    print(f"Speedup  (INT8 vs FP32): {speedup:.2f}x faster")
    print(f"Size reduction: {size_reduction:.1f}% smaller")
    print(f"FP32 avg latency: {fp32['avg_ms']:.1f} ms")
    print(f"INT8 avg latency: {int8['avg_ms']:.1f} ms")