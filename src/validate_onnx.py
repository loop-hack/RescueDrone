
from pathlib import Path
import time
import statistics

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "yolo11n.pt"
ONNX_PATH = ROOT / "yolo11n.onnx"
IMAGE_DIR = ROOT / "data" / "images"

CONF = 0.35
NMS_IOU = 0.45
REPEATS = 20


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh))

    left = round((size - nw) / 2 - 0.1)
    right = round((size - nw) / 2 + 0.1)
    top = round((size - nh) / 2 - 0.1)
    bottom = round((size - nh) / 2 + 0.1)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


def onnx_detect(session, image):
    padded, scale, px, py = letterbox(image, 320)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]

    start = time.perf_counter()
    output = session.run(
        None, {session.get_inputs()[0].name: tensor}
    )[0][0].T
    elapsed = time.perf_counter() - start

    scores = output[:, 4]
    selected = scores >= CONF
    predictions = output[selected]
    scores = scores[selected]

    boxes = []
    for row in predictions:
        cx, cy, w, h = row[:4]
        x1 = max(0, min(image.shape[1], (cx - w / 2 - px) / scale))
        y1 = max(0, min(image.shape[0], (cy - h / 2 - py) / scale))
        x2 = max(0, min(image.shape[1], (cx + w / 2 - px) / scale))
        y2 = max(0, min(image.shape[0], (cy + h / 2 - py) / scale))
        boxes.append([x1, y1, x2 - x1, y2 - y1])

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores.astype(float).tolist(), CONF, NMS_IOU
        )
        for idx in np.array(indices).reshape(-1):
            x, y, w, h = boxes[int(idx)]
            detections.append({
                "box": [x, y, x + w, y + h],
                "confidence": float(scores[int(idx)]),
            })

    return detections, elapsed


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def get_pt(model, image, size):
    start = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=size,
        classes=[0],
        conf=CONF,
        iou=NMS_IOU,
        device="cpu",
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - start

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                "box": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf[0].cpu()),
            })
    return detections, elapsed


def compare_boxes(reference, candidate):
    pairs = []
    for i, a in enumerate(reference):
        for j, b in enumerate(candidate):
            pairs.append((iou(a["box"], b["box"]), i, j))

    pairs.sort(reverse=True)
    used_a, used_b, matched = set(), set(), []

    for score, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append(score)

    return matched, len(reference) - len(used_a), len(candidate) - len(used_b)


def main():
    model = YOLO(str(PT_PATH))
    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )

    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        # Warm up both runtimes.
        get_pt(model, image, 320)
        onnx_detect(session, image)

        pt320, pt_times = None, []
        onnx_results, onnx_times = None, []

        for _ in range(REPEATS):
            pt320, elapsed = get_pt(model, image, 320)
            pt_times.append(elapsed)
            onnx_results, elapsed = onnx_detect(session, image)
            onnx_times.append(elapsed)

        pt640, _ = get_pt(model, image, 640)
        overlaps, unmatched_pt, unmatched_onnx = compare_boxes(
            pt320, onnx_results
        )

        print(f"\n{path.name}")
        print(f"  PyTorch 640 detections: {len(pt640)}")
        print(f"  PyTorch 320 detections: {len(pt320)}")
        print(f"  ONNX 320 detections:    {len(onnx_results)}")
        print(
            f"  PyTorch 320 median: {statistics.median(pt_times):.4f}s"
        )
        print(
            f"  ONNX 320 median:    {statistics.median(onnx_times):.4f}s"
        )
        print(
            f"  Box matches at IoU >= 0.50: "
            f"{sum(v >= 0.50 for v in overlaps)}/{max(len(pt320), len(onnx_results))}"
        )
        print(
            f"  Mean matched-box IoU: "
            f"{statistics.mean(overlaps):.3f}" if overlaps else
            "  Mean matched-box IoU: no matches"
        )
        print(
            f"  Unmatched PyTorch / ONNX boxes: "
            f"{unmatched_pt} / {unmatched_onnx}"
        )

    print("\nComparison complete. This is runtime parity, not ground-truth accuracy.")


if __name__ == "__main__":
    main()
