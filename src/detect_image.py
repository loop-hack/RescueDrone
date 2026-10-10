
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Detect people in a rescue image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    start = time.perf_counter()

    results = model.predict(
        source=image,
        classes=[0],  # COCO class 0 = person
        conf=args.conf,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    elapsed = time.perf_counter() - start
    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            detections.append({
                "class": "person",
                "confidence": round(
                    float(box.conf[0].cpu()), 4
                ),
                "bbox_xyxy": [x1, y1, x2, y2],
            })

    annotated = result.plot()
    image_output = output_dir / f"{image_path.stem}_detected.jpg"

    if not cv2.imwrite(str(image_output), annotated):
        raise SystemExit(f"Could not save image: {image_output}")

    report = {
        "source_image": str(image_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "confidence_threshold": args.conf,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "inference_seconds": round(elapsed, 4),
        "person_detections": len(detections),
        "detections": detections,
        "note": (
            "Detections require human verification. "
            "They do not establish whether a person is alive "
            "or needs rescue."
        ),
    }

    report_output = output_dir / f"{image_path.stem}_report.json"
    report_output.write_text(json.dumps(report, indent=2))

    print(f"\nImage size: {image.shape[1]} x {image.shape[0]}")
    print(f"People detected: {len(detections)}")
    print(f"Inference time: {elapsed:.3f} seconds")
    print(f"Annotated image: {image_output}")
    print(f"Evidence report: {report_output}")

    for index, detection in enumerate(detections, start=1):
        print(
            f"Person {index}: "
            f"confidence={detection['confidence']:.2f}, "
            f"box={detection['bbox_xyxy']}"
        )


if __name__ == "__main__":
    main()
