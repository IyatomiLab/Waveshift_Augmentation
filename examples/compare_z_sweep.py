"""How the WS1 effect grows with propagation distance z.

Writes a contact sheet and prints RMS deviation per z.
Run:  python examples/compare_z_sweep.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from waveshift import waveshift

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "_output"
Z_VALUES = [1, 5, 10, 20, 30, 41]
TILE = 192


def main():
    OUT.mkdir(exist_ok=True)
    image = Image.open(ROOT / "test.JPG").convert("RGB").resize((512, 512))
    reference = np.asarray(image, float)

    tiles = []
    print(f"{'z':>5}  {'RMS change':>10}")
    for z in Z_VALUES:
        augmented = waveshift(image, version="ws1", z=z)
        rms = np.sqrt(np.mean((np.asarray(augmented, float) - reference) ** 2))
        print(f"{z:>5}  {rms:>10.3f}")
        tiles.append(augmented.resize((TILE, TILE)))

    sheet = Image.new("RGB", (TILE * len(tiles), TILE))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * TILE, 0))
    sheet.save(OUT / "z_sweep.png")
    print(f"\nsaved {OUT / 'z_sweep.png'}  (z = {Z_VALUES})")
    print("note: the effect of a given z depends on image resolution;")
    print("      these numbers are for 512x512 input.")


if __name__ == "__main__":
    main()
