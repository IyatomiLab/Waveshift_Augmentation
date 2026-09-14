"""Deterministic Waveshift: fixed z and R, plus legacy reproduction.

Run:  python examples/deterministic_usage.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from waveshift import WaveShift, waveshift

ROOT = Path(__file__).resolve().parent.parent


def main():
    image = Image.open(ROOT / "test.JPG").convert("RGB").resize((512, 512))

    # 1. Fixed parameters: identical every time, no seed needed.
    a = waveshift(image, version="ws2", z=41, aperture=0.01)
    b = waveshift(image, version="ws2", z=41, aperture=0.01)
    print(f"fixed (z=41, R=0.01) reproducible : {np.array_equal(np.asarray(a), np.asarray(b))}")

    # 2. Sampled parameters with a seed: reproducible across processes.
    kw = dict(version="ws1", z_range=(1, 41), seed=123)
    c = WaveShift(**kw)(image)
    d = WaveShift(**kw)(image)
    print(f"seeded z_range reproducible       : {np.array_equal(np.asarray(c), np.asarray(d))}")

    # 3. The modern API never reads global RNG state.
    import random

    random.seed(0)
    np.random.seed(0)
    e = WaveShift(**kw)(image)
    print(f"unaffected by global RNG          : {np.array_equal(np.asarray(c), np.asarray(e))}")

    # 4. Reproduce the published GitHub snapshot exactly.
    legacy = waveshift(image, version="ws1", z=20.0, compatibility="legacy")
    modern = waveshift(image, version="ws1", z=20.0)
    diff = np.abs(np.asarray(legacy, int) - np.asarray(modern, int))
    print(f"legacy vs modern mean abs diff    : {diff.mean():.4f}")
    print("  (legacy keeps the off-centre grid and uint8 wraparound)")


if __name__ == "__main__":
    main()
