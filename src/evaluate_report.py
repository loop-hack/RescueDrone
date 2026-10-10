
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/evaluate_report.py outputs/bus_report.json")
        raise SystemExit(1)

    report_path = Path(sys.argv[1])

    if not report_path.is_file():
        raise SystemExit(f"Report not found: {report_path}")

    report = json.loads(report_path.read_text())

    width = report["image_width"]
    height = report["image_height"]
    detections = report["detections"]

    errors = []

    if report["person_detections"] != len(detections):
        errors.append("Detection count does not match detection list.")

    for index, detection in enumerate(detections, start=1):
        box = detection["bbox_xyxy"]

        if len(box) != 4:
            errors.append(f"Detection {index}: expected 4 box coordinates.")
            continue

        x1, y1, x2, y2 = box
        confidence = detection["confidence"]

        if not (0 <= x1 < x2 <= width):
            errors.append(f"Detection {index}: invalid horizontal coordinates.")

        if not (0 <= y1 < y2 <= height):
            errors.append(f"Detection {index}: invalid vertical coordinates.")

        if not (0 <= confidence <= 1):
            errors.append(f"Detection {index}: invalid confidence score.")

    print("=== RescueDrone Report Validation ===")
    print(f"Image dimensions: {width} x {height}")
    print(f"Detections: {len(detections)}")
    print(f"Inference time: {report['inference_seconds']:.4f} seconds")

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION PASSED")
    print("All bounding boxes, confidence scores, and counts are structurally valid.")
    print("This does not measure detection accuracy.")


if __name__ == "__main__":
    main()
