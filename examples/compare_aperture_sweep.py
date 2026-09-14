"""How the WS2 effect grows with the aperture coefficient R.

Writes a contact sheet and prints RMS deviation per R, at fixed z.
Run:  python examples/compare_aperture_sweep.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from waveshift import waveshift

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "_output"
APERTURES = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
Z = 20.0
TILE = 192


def main():
    OUT.mkdir(exist_ok=True)
    image = Image.open(ROOT / "test.JPG").convert("RGB").resize((512, 512))
    reference = np.asarray(image, float)

    tiles = []
    print(f"fixed z = {Z}\n")
    print(f"{'R':>8}  {'RMS change':>10}  {'first-zero radius (px)':>24}")
    for aperture in APERTURES:
        augmented = waveshift(image, version="ws2", z=Z, aperture=aperture)
        rms = np.sqrt(np.mean((np.asarray(augmented, float) - reference) ** 2))
        # First zero of the Airy pattern sits at R*r = 3.8317.
        print(f"{aperture:>8}  {rms:>10.3f}  {3.8317 / aperture:>24.1f}")
        tiles.append(augmented.resize((TILE, TILE)))

    sheet = Image.new("RGB", (TILE * len(tiles), TILE))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * TILE, 0))
    sheet.save(OUT / "aperture_sweep.png")
    print(f"\nsaved {OUT / 'aperture_sweep.png'}  (R = {APERTURES})")
    print("half-diagonal of a 512x512 spectrum is ~362 px: R=0.01 places the")
    print("Airy main lobe just around the image spectrum.")


if __name__ == "__main__":
    main()
