"""WS2: fixed z, varying aperture coefficient R.

Top row is the augmented result; bottom row is the Airy envelope |q| actually
applied in the Fourier domain, which is what makes the attenuation legible --
as R grows the passband shrinks and the image smooths.

Concept adapted from `psf_airy_disk.m` and `Propagator_Re_Power.m`.

Run:  python scripts/generate_assets/generate_ws2_aperture_sweep.py
"""

import numpy as np
import matplotlib.pyplot as plt

from waveshift import waveshift, ws2_propagator
from waveshift.propagators import WAVELENGTHS_RGB

from _common import SIZE, Z_DEMO, save, sample_image, show, use_style

APERTURES = [0.0001, 0.001, 0.01, 0.05]


def main():
    use_style()
    image = sample_image()
    reference = np.asarray(image, dtype=float)

    fig, axes = plt.subplots(2, len(APERTURES), figsize=(10.4, 5.6))
    for column, aperture in enumerate(APERTURES):
        augmented = waveshift(image, version="ws2", z=Z_DEMO, aperture=aperture)
        rms = float(np.sqrt(np.mean((np.asarray(augmented, float) - reference) ** 2)))
        show(axes[0, column], augmented, f"R = {aperture:g}\nRMS {rms:.1f}")

        envelope = np.abs(
            ws2_propagator((SIZE, SIZE), WAVELENGTHS_RGB[1], Z_DEMO, aperture)
        )
        show(axes[1, column], envelope, cmap="viridis", vmin=0, vmax=1)
        first_zero = 3.8317 / aperture
        axes[1, column].set_xlabel(
            f"passband radius ≈ {min(first_zero, 999999):.0f} px", fontsize=8
        )

    axes[1, 0].set_ylabel("Airy envelope |q|", fontsize=9)
    fig.suptitle(
        "WS2: the aperture coefficient R sets an Airy envelope in the Fourier "
        "domain. Larger R = narrower passband = stronger smoothing.",
        fontsize=11, y=0.97,
    )
    fig.text(0.5, 0.015,
             f"Fixed z = {Z_DEMO:g}, 512x512 input. The half-diagonal of this "
             "spectrum is ~362 px, so R ≈ 0.01 is the point where the envelope "
             "starts to bite.",
             ha="center", fontsize=8.5, color="#666666")
    fig.subplots_adjust(wspace=0.08, hspace=0.02, top=0.87, bottom=0.08)
    save(fig, "ws2_aperture_sweep.png")


if __name__ == "__main__":
    main()
