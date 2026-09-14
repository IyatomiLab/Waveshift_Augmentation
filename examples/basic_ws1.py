"""WS1: phase-only Fresnel propagation (IEEE Access 2025).

Run:  python examples/basic_ws1.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from waveshift import WaveShift

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "_output"


def main():
    OUT.mkdir(exist_ok=True)
    image = Image.open(ROOT / "test.JPG").convert("RGB").resize((512, 512))

    # z is sampled per call from z_range; seed makes that reproducible.
    transform = WaveShift(version="ws1", z_range=(1, 41), seed=0)
    augmented = transform(image)

    augmented.save(OUT / "ws1.png")
    before, after = np.asarray(image, float), np.asarray(augmented, float)
    print(f"input     {image.size} {image.mode}")
    print(f"output    {augmented.size} {augmented.mode}")
    print(f"RMS change {np.sqrt(np.mean((after - before) ** 2)):.3f}")
    print(f"saved     {OUT / 'ws1.png'}")


if __name__ == "__main__":
    main()
