"""README hero: the whole method in one figure.

    original image -> Fourier spectrum -> WS propagator -> augmented image

Concept adapted from the thesis figure script `Example_visual_schematic.m`,
rewritten to use the packaged API only.

Run:  python scripts/generate_assets/generate_hero_concept.py
"""

import numpy as np
import matplotlib.pyplot as plt

from waveshift import waveshift, ws1_propagator
from waveshift.propagators import WAVELENGTHS_RGB

from _common import SIZE, Z_DEMO, log_spectrum, save, sample_image, show, use_style


def main():
    use_style()
    image = sample_image()
    green = np.asarray(image)[..., 1].astype(float)

    spectrum = log_spectrum(green)
    # WS1 here: its concentric Fresnel rings read clearly at thumbnail size,
    # whereas the WS2 Airy envelope damps them into a featureless blob.
    propagator = ws1_propagator((SIZE, SIZE), WAVELENGTHS_RGB[1], Z_DEMO)
    augmented = waveshift(image, version="ws1", z=Z_DEMO)
    rms = float(
        np.sqrt(np.mean((np.asarray(augmented, float) - np.asarray(image, float)) ** 2))
    )

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5))

    show(axes[0], image, "1. Input image")
    show(axes[1], spectrum, "2. Fourier spectrum\n(log magnitude)", cmap="magma")
    show(axes[2], np.real(propagator),
         f"3. Waveshift propagator\n(WS1, z = {Z_DEMO:g})",
         cmap="RdBu_r", vmin=-1, vmax=1)
    show(axes[3], augmented, f"4. Augmented image\n(RMS change {rms:.1f}/255)")

    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.14)
    # Place the arrows from the laid-out axes rather than guessed coordinates.
    for left, right in zip(axes[:-1], axes[1:]):
        a, b = left.get_position(), right.get_position()
        fig.text((a.x1 + b.x0) / 2, (a.y0 + a.y1) / 2, "→",
                 ha="center", va="center", fontsize=20, color="#555555")

    fig.suptitle(
        "Waveshift: propagate the image spectrum to a nearby wavefront, "
        "then render what the camera would see",
        fontsize=11.5, y=0.95,
    )
    fig.text(
        0.5, 0.025,
        "Shown on a plant leaf; the method is image-agnostic and applies to any "
        "fine-grained classification task. The change is deliberately subtle -- "
        "it is a physically plausible re-imaging, not a distortion.",
        ha="center", fontsize=8.5, color="#666666",
    )
    save(fig, "waveshift_hero.png")


if __name__ == "__main__":
    main()
