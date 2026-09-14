# Minimal runnability check. Verifies that the API executes on the
# supported input types and that legacy reproduction still works.

import numpy as np
from PIL import Image

from waveshift import WaveShift, apply_to_regions, waveshift

IMG_PATH = "test.JPG"
SIZE = (512, 512)


def describe(out):
    array = np.asarray(out)
    kind = f"PIL {out.mode} {out.size}" if isinstance(out, Image.Image) else "ndarray"
    return f"{kind} -> {array.shape} {array.dtype} [{array.min()}, {array.max()}]"


def check(label, out, expected_shape):
    array = np.asarray(out)
    assert array.shape == expected_shape, f"{label}: {array.shape} != {expected_shape}"
    assert array.dtype == np.uint8, f"{label}: dtype {array.dtype}"
    print(f"  {label:32s} OK  {describe(out)}")


def main():
    print("Waveshift smoke test")
    leaf = Image.open(IMG_PATH).convert("RGB").resize(SIZE)
    rect = Image.open(IMG_PATH).convert("RGB").resize((640, 480))

    check("modern ws1 RGB square", WaveShift(version="ws1", z=20)(leaf), (512, 512, 3))
    check("modern ws2 RGB square",
          WaveShift(version="ws2", z=20, aperture=0.01)(leaf), (512, 512, 3))
    check("modern ws1 RGB rectangular",
          WaveShift(version="ws1", z=20)(rect), (480, 640, 3))
    check("modern ws1 grayscale",
          WaveShift(version="ws1", z=20)(rect.convert("L")), (480, 640))
    check("modern ws1 numpy array",
          WaveShift(version="ws1", z=20)(np.asarray(rect)), (480, 640, 3))
    check("legacy ws1 (published path)",
          waveshift(leaf, version="ws1", z=20, compatibility="legacy"), (512, 512, 3))
    check("ws2.5 ROI",
          apply_to_regions(leaf, [(50, 50, 300, 280)], version="ws2", z=20,
                           aperture=0.01), (512, 512, 3))

    print("PASS")


if __name__ == "__main__":
    main()
