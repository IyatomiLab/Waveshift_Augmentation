"""WS2.5: apply Waveshift only inside regions of interest.

This is the PhD thesis extension (Appendix G), NOT part of either journal
paper. Boxes come from the caller; no detector is required.

Run:  python examples/roi_usage.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from waveshift import apply_to_regions

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "_output"


def yolo_boxes(image_path):
    """Optional: derive boxes from YOLOv8.

    ultralytics is NOT a dependency of this package. This helper is here to
    show the shape of the integration; it is never called by default.
    """
    from ultralytics import YOLO  # noqa: F401  (optional, not installed)

    model = YOLO("yolov8n.pt")
    result = model(str(image_path))[0]
    return [tuple(map(int, box)) for box in result.boxes.xyxy.tolist()]


def main():
    OUT.mkdir(exist_ok=True)
    image = Image.open(ROOT / "test.JPG").convert("RGB").resize((512, 512))

    # Hand-specified boxes: (left, top, right, bottom).
    boxes = [(60, 50, 250, 240), (280, 270, 470, 460)]

    hard = apply_to_regions(image, boxes, version="ws2", z=41, aperture=0.01)
    soft = apply_to_regions(
        image, boxes, version="ws2", z=41, aperture=0.01, feather=16
    )

    source = np.asarray(image)
    for name, result in (("hard", hard), ("soft", soft)):
        array = np.asarray(result)
        untouched = np.ones(source.shape[:2], bool)
        for left, top, right, bottom in boxes:
            untouched[top:bottom, left:right] = False
        print(f"{name:5s} outside ROI unchanged: {np.array_equal(array[untouched], source[untouched])}")
        result.save(OUT / f"roi_{name}.png")

    print(f"\nsaved {OUT / 'roi_hard.png'} and {OUT / 'roi_soft.png'}")
    print("note: a given z acts more weakly on a small ROI than on a full")
    print("      frame; pass scale_z='area' to compensate.")


if __name__ == "__main__":
    main()
