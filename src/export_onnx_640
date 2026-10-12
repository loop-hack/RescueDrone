
from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "yolo11n.pt"
ONNX_320 = ROOT / "yolo11n.onnx"
ONNX_320_BACKUP = ROOT / "yolo11n_320.onnx"
ONNX_640 = ROOT / "yolo11n_640.onnx"


def main():
    if not PT_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {PT_PATH}")

    # Preserve the validated 320 model before exporting another resolution.
    if ONNX_320.is_file() and not ONNX_320_BACKUP.exists():
        shutil.copy2(ONNX_320, ONNX_320_BACKUP)
        print(f"Preserved 320 model: {ONNX_320_BACKUP}")

    print("Exporting YOLO11n at 640 x 640...")
    model = YOLO(str(PT_PATH))

    exported = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=False,
        half=False,
        device="cpu",
    )

    exported_path = Path(exported)

    if exported_path.resolve() != ONNX_640.resolve():
        if ONNX_640.exists():
            ONNX_640.unlink()
        shutil.move(str(exported_path), str(ONNX_640))

    print("\n=== Export complete ===")
    print(f"320 model preserved: {ONNX_320_BACKUP.exists()}")
    print(f"640 model: {ONNX_640}")
    print(f"640 model size: {ONNX_640.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
