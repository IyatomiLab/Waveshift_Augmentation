# smoke_test.py
# Minimal runnability check for Waveshift.
# Verifies only that both modes execute and return a well-formed image.
# Does NOT assert any scientific/numerical behavior.

import numpy as np
from PIL import Image

from Waveshift import Wavefront_Shift

IMG_PATH = "test.JPG"
SIZE = (512, 512)


def check(mode_):
    leaf = Image.open(IMG_PATH).convert("RGB").resize(SIZE)
    out = Wavefront_Shift(mode_=mode_)(leaf)

    assert isinstance(out, Image.Image), f"{mode_}: not a PIL image"
    assert out.mode == "RGB", f"{mode_}: mode is {out.mode}, expected RGB"
    assert out.size == SIZE, f"{mode_}: size is {out.size}, expected {SIZE}"

    arr = np.array(out)
    assert arr.dtype == np.uint8, f"{mode_}: dtype is {arr.dtype}, expected uint8"
    assert arr.shape == (SIZE[1], SIZE[0], 3), f"{mode_}: shape is {arr.shape}"

    print(f"  mode_={mode_!r:6s} OK  size={out.size} mode={out.mode} "
          f"dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")


if __name__ == "__main__":
    print("Waveshift smoke test")
    check("s")      # WS 1.0
    check("psf")    # WS 2.0
    print("PASS")
