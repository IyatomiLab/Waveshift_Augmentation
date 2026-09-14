"""WS2: WS1 phase plus an Airy-disk aperture envelope (Electronics 2025).

Run:  python examples/basic_ws2.py
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

    # WS2 adds the aperture coefficient R. Smaller R -> gentler low-pass.
    transform = WaveShift(
        version="ws2", z_range=(15, 151), aperture_range=(0.0001, 0.01), seed=0
    )
    augmented = transform(image)

    augmented.save(OUT / "ws2.png")
    before, after = np.asarray(image, float), np.asarray(augmented, float)
    print(f"output    {augmented.size} {augmented.mode}")
    print(f"RMS change {np.sqrt(np.mean((after - before) ** 2)):.3f}")
    print(f"saved     {OUT / 'ws2.png'}")


if __name__ == "__main__":
    main()
