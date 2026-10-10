
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "benchmark.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    model = YOLO("yolo11n.pt")
    rows = []

    # Warm up the model; exclude this time from the measurements.
    model.predict(
        source=str(image_paths[0]),
        classes=[0],
        conf=0.35,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        start = time.perf_counter()

        result = model.predict(
            source=image,
            classes=[0],
            conf=0.35,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        elapsed = time.perf_counter() - start

        boxes = result.boxes
        confidences = (
            [round(float(value), 4) for value in boxes.conf.cpu().tolist()]
            if boxes is not None
            else []
        )

        row = {
            "image": image_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "people_detected": len(confidences),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 4)
                if confidences else 0
            ),
            "inference_seconds": round(elapsed, 4),
        }
        rows.append(row)

        print(
            f"{row['image']}: {row['people_detected']} detections, "
            f"{row['inference_seconds']:.3f}s"
        )

    if not rows:
        raise SystemExit("No readable images could be evaluated.")

    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    mean_time = sum(row["inference_seconds"] for row in rows) / len(rows)

    print(f"\nImages evaluated: {len(rows)}")
    print(f"Mean inference time: {mean_time:.3f}s")
    print(f"Benchmark CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()