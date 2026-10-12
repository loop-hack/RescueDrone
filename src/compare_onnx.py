
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "yolo11n.pt"
ONNX_PATH = ROOT / "yolo11n.onnx"
IMAGE_DIR = ROOT / "data" / "images"

IMAGE_SIZE = 320
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


def letterbox(image, size=320):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height))

    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2

    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, left, top


def run_onnx(session, image):
    padded, scale, pad_x, pad_y = letterbox(image, IMAGE_SIZE)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    output = session.run(None, {input_name: tensor})[0]
    elapsed = time.perf_counter() - start

    # YOLO11 output: [1, 84, candidates]
    predictions = output[0].T

    # First four values are xywh; remaining 80 are class scores.
    class_scores = predictions[:, 4:]
    person_scores = class_scores[:, 0]

    selected = person_scores >= CONF_THRESHOLD
    predictions = predictions[selected]
    scores = person_scores[selected]

    if len(predictions) == 0:
        return [], elapsed

    boxes = []
    for pred in predictions:
        cx, cy, width, height = pred[:4]

        x1 = (cx - width / 2 - pad_x) / scale
        y1 = (cy - height / 2 - pad_y) / scale
        x2 = (cx + width / 2 - pad_x) / scale
        y2 = (cy + height / 2 - pad_y) / scale

        x1 = max(0, min(image.shape[1], x1))
        y1 = max(0, min(image.shape[0], y1))
        x2 = max(0, min(image.shape[1], x2))
        y2 = max(0, min(image.shape[0], y2))

        boxes.append([
            float(x1), float(y1),
            float(x2 - x1), float(y2 - y1),
        ])

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores.astype(float).tolist(),
        CONF_THRESHOLD,
        IOU_THRESHOLD,
    )

    detections = []
    if len(indices):
        for index in np.array(indices).reshape(-1):
            x, y, w, h = boxes[int(index)]
            detections.append({
                "confidence": float(scores[int(index)]),
                "xyxy": [x, y, x + w, y + h],
            })

    return detections, elapsed


def main():
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {ONNX_PATH}")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    pt_model = YOLO(str(PT_PATH))

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No test images in {IMAGE_DIR}")

    print(f"Image size: {IMAGE_SIZE}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"NMS IoU threshold: {IOU_THRESHOLD}")

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"\nSkipping unreadable image: {path.name}")
            continue

        # Warm up both runtimes; exclude warm-up from reported timings.
        run_onnx(session, image)
        pt_model.predict(
            source=image, imgsz=IMAGE_SIZE, classes=[0],
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            device="cpu", verbose=False,
        )

        onnx_detections, onnx_time = run_onnx(session, image)

        start = time.perf_counter()
        pt_result = pt_model.predict(
            source=image, imgsz=IMAGE_SIZE, classes=[0],
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            device="cpu", verbose=False,
        )[0]
        pt_time = time.perf_counter() - start

        pt_count = len(pt_result.boxes) if pt_result.boxes is not None else 0

        print(f"\n{path.name} ({image.shape[1]}x{image.shape[0]})")
        print(f"  PyTorch detections: {pt_count}, time: {pt_time:.4f}s")
        print(
            f"  ONNX detections:   {len(onnx_detections)}, "
            f"time: {onnx_time:.4f}s"
        )
        print(
            "  Count match:",
            pt_count == len(onnx_detections),
        )


if __name__ == "__main__":
    main()
