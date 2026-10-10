
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
GT_PATH = ROOT / "data" / "ground_truth.json"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50


def iou(box_a, box_b):
    """Intersection over Union for [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def match_boxes(predictions, ground_truth):
    """Greedy one-to-one matching, highest IoU first."""
    candidates = []

    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            overlap = iou(pred, gt)
            if overlap >= IOU_THRESHOLD:
                candidates.append((overlap, pred_idx, gt_idx))

    candidates.sort(reverse=True)

    matched_predictions = set()
    matched_ground_truth = set()

    for _, pred_idx, gt_idx in candidates:
        if pred_idx not in matched_predictions and gt_idx not in matched_ground_truth:
            matched_predictions.add(pred_idx)
            matched_ground_truth.add(gt_idx)

    tp = len(matched_predictions)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def main():
    if not GT_PATH.exists():
        sys.exit(f"Ground truth not found: {GT_PATH}")

    ground_truth_data = json.loads(GT_PATH.read_text())
    if not ground_truth_data:
        sys.exit("Ground-truth file is empty.")

    model = YOLO(MODEL_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_tp = total_fp = total_fn = 0
    results = []

    for filename, annotation in ground_truth_data.items():
        image_path = IMAGE_DIR / filename
        image = cv2.imread(str(image_path))

        if image is None:
            sys.exit(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        if width != annotation["width"] or height != annotation["height"]:
            sys.exit(f"Image dimensions do not match ground truth: {filename}")

        gt_boxes = annotation["people"]

        result = model.predict(
            source=image,
            classes=[0],
            conf=CONFIDENCE,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        pred_boxes = result.boxes.xyxy.cpu().tolist()
        pred_boxes = [[float(v) for v in box] for box in pred_boxes]

        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        results.append({
            "image": filename,
            "ground_truth": len(gt_boxes),
            "predictions": len(pred_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
        })

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    report = {
        "model": MODEL_PATH,
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU_THRESHOLD,
        "images_evaluated": len(results),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_image": results,
        "note": "Results apply only to the annotated images and settings.",
    }

    report_path = OUTPUT_DIR / "accuracy_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Model: {MODEL_PATH}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Confidence threshold: {CONFIDENCE}\n")

    for item in results:
        print(
            f"{item['image']}: GT={item['ground_truth']} "
            f"Pred={item['predictions']} TP={item['TP']} "
            f"FP={item['FP']} FN={item['FN']} "
            f"Precision={item['precision']:.3f} "
            f"Recall={item['recall']:.3f}"
        )

    print(f"\nOverall: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()