"""WS1 vs WS2 propagators as simple 2-D panels.

WS1 is pure phase (|p| = 1 everywhere); WS2 multiplies that phase by an Airy
intensity envelope, so its amplitude falls away from the centre.

Concept adapted from `Propagator_Re_Power.m` and
`Propagator_Amplitude_Phase.m`, rendered as flat panels rather than 3-D
surfaces so they stay readable at README size.

Run:  python scripts/generate_assets/generate_propagator_panels.py
"""

import numpy as np
import matplotlib.pyplot as plt

from waveshift import ws1_propagator, ws2_propagator
from waveshift.propagators import WAVELENGTHS_RGB

from _common import APERTURE_DEMO, SIZE, Z_DEMO, save, show, use_style

WAVELENGTH = WAVELENGTHS_RGB[1]


def main():
    use_style()
    shape = (SIZE, SIZE)
    ws1 = ws1_propagator(shape, WAVELENGTH, Z_DEMO)
    ws2 = ws2_propagator(shape, WAVELENGTH, Z_DEMO, APERTURE_DEMO)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))

    real_kw = dict(cmap="RdBu_r", vmin=-1, vmax=1)
    amp_kw = dict(cmap="viridis", vmin=0, vmax=1)

    h1 = show(axes[0, 0], np.real(ws1), "WS1 — real part", **real_kw)
    h2 = show(axes[0, 1], np.abs(ws1), "WS1 — amplitude  (uniform = 1)", **amp_kw)
    show(axes[1, 0], np.real(ws2), "WS2 — real part", **real_kw)
    show(axes[1, 1], np.abs(ws2), "WS2 — amplitude  (Airy envelope)", **amp_kw)

    fig.colorbar(h1, ax=axes[:, 0], shrink=0.6, location="bottom", pad=0.04)
    fig.colorbar(h2, ax=axes[:, 1], shrink=0.6, location="bottom", pad=0.04)

    fig.suptitle(
        f"Propagators at z = {Z_DEMO:g}, R = {APERTURE_DEMO:g}\n"
        "WS1 is phase-only and energy-preserving; WS2 also attenuates amplitude.",
        fontsize=11, y=0.965,
    )
    save(fig, "propagator_ws1_ws2.png")


if __name__ == "__main__":
    main()
