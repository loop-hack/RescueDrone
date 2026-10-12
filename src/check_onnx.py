from importlib.util import find_spec
import platform
import sys

print("Python:", sys.version.split()[0])
print("Architecture:", platform.machine())

for package in ("onnx", "onnxruntime", "onnxscript"):
    print(f"{package} installed:", find_spec(package) is not None)
