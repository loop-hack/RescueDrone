
from pathlib import Path
import time

import cv2
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolo11n.onnx"
IMAGE_PATH = ROOT / "data" / "images" / "bus.jpg"

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    print("ONNX Runtime:", ort.__version__)
    print("Model size:", round(MODEL_PATH.stat().st_size / (1024 * 1024), 2), "MiB")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("Providers:", session.get_providers())

    for item in session.get_inputs():
        print("Input:", item.name, item.shape, item.type)

    for item in session.get_outputs():
        print("Output:", item.name, item.shape, item.type)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    print("Image dimensions:", image.shape[1], "x", image.shape[0])

    print("\nONNX session loaded successfully.")
    print("Next we will implement matching preprocessing and compare detections.")

if __name__ == "__main__":
    main()