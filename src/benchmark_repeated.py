import csv
import math
import resource
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolo11n.pt"
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "benchmark_repeated.csv"

WARMUP_RUNS = 3
MEASURED_RUNS = 20
CONFIDENCE = 0.35
IMAGE_SIZE = 640


def percentile(values, percent):
    ordered = sorted(values)
    index = math.ceil(percent / 100 * len(ordered)) - 1
    return ordered[max(0, index)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    model_bytes = MODEL_PATH.stat().st_size
    print(f"Model: {MODEL_PATH.name} ({model_bytes / 1024**2:.2f} MiB)")
    print(f"Warm-up runs per image: {WARMUP_RUNS}")
    print(f"Measured runs per image: {MEASURED_RUNS}")
    print("Device: CPU")
    print("\nBenchmarking...")

    model = YOLO(str(MODEL_PATH))
    rows = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        def infer():
            return model.predict(
                source=image,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )[0]

        for _ in range(WARMUP_RUNS):
            infer()

        timings = []
        last_result = None

        for _ in range(MEASURED_RUNS):
            start = time.perf_counter()
            last_result = infer()
            timings.append(time.perf_counter() - start)

        boxes = last_result.boxes
        detections = len(boxes) if boxes is not None else 0

        row = {
            "image": path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "runs": len(timings),
            "detections_last_run": detections,
            "median_seconds": round(statistics.median(timings), 5),
            "p95_seconds": round(percentile(timings, 95), 5),
            "mean_seconds": round(statistics.mean(timings), 5),
            "approx_fps_single_image": round(
                1 / statistics.mean(timings), 2
            ),
        }
        rows.append(row)

        print(
            f"{path.name}: median={row['median_seconds']:.4f}s, "
            f"p95={row['p95_seconds']:.4f}s, "
            f"approx={row['approx_fps_single_image']:.2f} FPS, "
            f"detections={detections}"
        )

    if not rows:
        raise SystemExit("No readable images were benchmarked.")

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Linux reports ru_maxrss in KiB.
    peak_rss_mib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024

    print("\n=== Summary ===")
    print(f"Images: {len(rows)}")
    print(f"Model file: {model_bytes / 1024**2:.2f} MiB")
    print(f"Process peak RSS: {peak_rss_mib:.1f} MiB")
    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()