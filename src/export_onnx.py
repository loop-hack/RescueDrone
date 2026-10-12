
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolo11n.pt"

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    exported_path = model.export(
        format="onnx",
        imgsz=320,
        batch=1,
        dynamic=False,
        simplify=False,
        half=False,
        device="cpu",
    )

    output = Path(exported_path)
    print("\n=== Export complete ===")
    print(f"ONNX model: {output}")
    print(f"File size: {output.stat().st_size / (1024 * 1024):.2f} MiB")
    print("Input resolution: 320 x 320")
    print("Batch size: 1")

if __name__ == "__main__":
    main()