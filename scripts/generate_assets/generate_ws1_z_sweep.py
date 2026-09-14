"""WS1: the same image propagated at increasing z.

Top row is the augmented result, bottom row the amplified difference from the
original -- at 512x512 the effect is a few grey levels, so it needs amplifying
to be visible in print.

Concept adapted from `PropagatorS_View.m` / `Data_Augmentation_RGB.py`.

Run:  python scripts/generate_assets/generate_ws1_z_sweep.py
"""

import numpy as np
import matplotlib.pyplot as plt

from waveshift import waveshift

from _common import save, sample_image, show, use_style

Z_VALUES = [1, 11, 21, 31, 41]
GAIN = 6


def main():
    use_style()
    image = sample_image()
    reference = np.asarray(image, dtype=float)

    fig, axes = plt.subplots(2, len(Z_VALUES), figsize=(12.2, 5.3))
    for column, z in enumerate(Z_VALUES):
        augmented = waveshift(image, version="ws1", z=float(z))
        array = np.asarray(augmented, dtype=float)
        rms = float(np.sqrt(np.mean((array - reference) ** 2)))

        show(axes[0, column], augmented, f"z = {z}")
        difference = np.clip(np.abs(array - reference) * GAIN, 0, 255) / 255.0
        show(axes[1, column], difference, f"|difference| x{GAIN}   RMS {rms:.1f}")

    axes[0, 0].set_ylabel("augmented", fontsize=9)
    axes[1, 0].set_ylabel("difference", fontsize=9)
    fig.suptitle(
        "WS1: phase-only propagation. Larger z shifts the wavefront further, "
        "redistributing detail between colour channels.",
        fontsize=11, y=0.97,
    )
    fig.text(0.5, 0.015,
             "512x512 input. z is in model units, not calibrated metres; its "
             "effect depends on image resolution.",
             ha="center", fontsize=8.5, color="#666666")
    fig.subplots_adjust(wspace=0.06, hspace=0.16, top=0.86, bottom=0.08)
    save(fig, "ws1_z_sweep.png")


if __name__ == "__main__":
    main()
