
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT = ROOT / "data" / "ground_truth.json"

ground_truth = {}

for xml_file in sorted(IMAGE_DIR.glob("*.xml")):
    root = ET.parse(xml_file).getroot()
    filename = root.findtext("filename")
    size = root.find("size")

    people = []

    for obj in root.findall("object"):
        if obj.findtext("name") != "person":
            continue

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_file}")

        x1 = int(box.findtext("xmin"))
        y1 = int(box.findtext("ymin"))
        x2 = int(box.findtext("xmax"))
        y2 = int(box.findtext("ymax"))

        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"Invalid box in {xml_file}: {(x1, y1, x2, y2)}"
            )

        people.append([x1, y1, x2, y2])

    ground_truth[filename] = {
        "width": int(size.findtext("width")),
        "height": int(size.findtext("height")),
        "people": people,
    }

OUTPUT.write_text(json.dumps(ground_truth, indent=2) + "\n")
print(f"Saved {OUTPUT}")

for filename, data in ground_truth.items():
    print(f"{filename}: {len(data['people'])} annotated people")