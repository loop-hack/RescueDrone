
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "yolo11n.pt"

MODELS = {
    320: ROOT / "yolo11n_320.onnx",
    640: ROOT / "yolo11n_640.onnx",
}

IMAGES = [
    ROOT / "data/images/bus.jpg",
    ROOT / "data/images/zidane.jpg",
]

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
WARMUP_RUNS = 5
MEASURED_RUNS = 20


def letterbox(image, size):
    """Resize and pad an image to a square, preserving aspect ratio."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)

    new_width = round(width * ratio)
    new_height = round(height * ratio)

    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    pad_width = size - new_width
    pad_height = size - new_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0

    return tensor[None, ...], ratio, left, top


def run_onnx(session, image, size):
    tensor, ratio, pad_x, pad_y = letterbox(image, size)

    outputs = session.run(
        None, {session.get_inputs()[0].name: tensor}
    )[0][0]

    # YOLO output: 4 box values + 80 class scores per candidate.
    boxes = outputs[:4, :].T
    scores = outputs[4:, :].T

    # Person is COCO class 0.
    person_scores = scores[:, 0]
    selected = person_scores >= CONF_THRESHOLD

    boxes = boxes[selected]
    person_scores = person_scores[selected]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    # Convert center-x, center-y, width, height to x1, y1, x2, y2.
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # OpenCV NMSBoxes expects x, y, width, height.
    nms_boxes = [
        [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ]
        for box in xyxy
    ]

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        person_scores.tolist(),
        CONF_THRESHOLD,
        IOU_THRESHOLD,
    )

    if len(indices) == 0:
        return np.empty((0, 4)), np.empty((0,))

    indices = np.asarray(indices).reshape(-1)
    xyxy = xyxy[indices]
    final_scores = person_scores[indices]

    # Undo letterboxing to recover original-image coordinates.
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio

    height, width = image.shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

    return xyxy, final_scores


def median_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_boxes(reference, candidate, threshold=0.50):
    """Greedy one-to-one matching by highest IoU."""
    pairs = []

    for i, box_a in enumerate(reference):
        for j, box_b in enumerate(candidate):
            score = median_iou(box_a, box_b)
            if score >= threshold:
                pairs.append((score, i, j))

    pairs.sort(reverse=True)
    used_a = set()
    used_b = set()
    matched_ious = []

    for score, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched_ious.append(score)

    return len(matched_ious), matched_ious


def benchmark(function):
    for _ in range(WARMUP_RUNS):
        function()

    timings = []
    result = None

    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)

    return result, np.asarray(timings)


def main():
    for path in [PT_PATH, *MODELS.values(), *IMAGES]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    print("Loading PyTorch model...")
    pytorch_model = YOLO(str(PT_PATH))

    sessions = {}
    for size, path in MODELS.items():
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        shape = session.get_inputs()[0].shape
        print(f"ONNX {size}: {path.name}, input shape={shape}")
        sessions[size] = session

    print(f"\nWarm-ups: {WARMUP_RUNS}; measured runs: {MEASURED_RUNS}")
    print(f"Confidence threshold: {CONF_THRESHOLD}; NMS IoU: {IOU_THRESHOLD}")

    for image_path in IMAGES:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\nCould not read image: {image_path}")
            continue

        print(f"\n{'=' * 60}\nImage: {image_path.name}")

        for size in (320, 640):
            session = sessions[size]

            def pytorch_run():
                result = pytorch_model.predict(
                    source=image,
                    imgsz=size,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=[0],
                    device="cpu",
                    verbose=False,
                )[0]

                if result.boxes is None:
                    return np.empty((0, 4))

                return result.boxes.xyxy.cpu().numpy()

            pt_boxes, pt_times = benchmark(pytorch_run)

            def onnx_run():
                boxes, _ = run_onnx(session, image, size)
                return boxes

            onnx_boxes, onnx_times = benchmark(onnx_run)

            count, ious = match_boxes(pt_boxes, onnx_boxes)
            mean_iou = float(np.mean(ious)) if ious else 0.0

            print(f"\nResolution: {size} x {size}")
            print(f"  PyTorch detections: {len(pt_boxes)}")
            print(f"  ONNX detections:    {len(onnx_boxes)}")
            print(
                f"  PyTorch median: {np.median(pt_times):.4f}s "
                f"(p95 {np.percentile(pt_times, 95):.4f}s)"
            )
            print(
                f"  ONNX median:    {np.median(onnx_times):.4f}s "
                f"(p95 {np.percentile(onnx_times, 95):.4f}s)"
            )
            print(
                f"  Matched boxes at IoU >= 0.50: "
                f"{count}/{max(len(pt_boxes), len(onnx_boxes))}"
            )
            print(f"  Mean matched-box IoU: {mean_iou:.3f}")


if __name__ == "__main__":
    main()
