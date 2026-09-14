"""Before/after grid: original vs WS1 vs WS2, with a zoomed detail row.

The audit found no asset anywhere in the repository that showed an actually
augmented image, only propagator surfaces. This fills that gap.

Run:  python scripts/generate_assets/generate_before_after_grid.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from waveshift import waveshift

from _common import APERTURE_DEMO, Z_DEMO, save, sample_image, show, use_style

CROP = (150, 150, 278, 278)  # left, top, right, bottom -- 128 px detail


def main():
    use_style()
    image = sample_image()
    ws1 = waveshift(image, version="ws1", z=Z_DEMO)
    ws2 = waveshift(image, version="ws2", z=Z_DEMO, aperture=APERTURE_DEMO)

    panels = [
        (image, "Original"),
        (ws1, f"WS1   z = {Z_DEMO:g}"),
        (ws2, f"WS2   z = {Z_DEMO:g}, R = {APERTURE_DEMO:g}"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for column, (panel, title) in enumerate(panels):
        show(axes[0, column], panel, title)
        left, top, right, bottom = CROP
        axes[0, column].add_patch(
            Rectangle((left, top), right - left, bottom - top,
                      fill=False, edgecolor="#e8a33d", linewidth=1.4)
        )
        show(axes[1, column], panel.crop(CROP))

    axes[1, 0].set_ylabel("detail (128 px)", fontsize=9)
    fig.suptitle(
        "Before and after. WS1 redistributes detail between channels; "
        "WS2 additionally softens high spatial frequencies.",
        fontsize=11, y=0.965,
    )
    fig.subplots_adjust(wspace=0.06, hspace=-0.12, top=0.90, bottom=0.02)
    save(fig, "ws_before_after.png")


if __name__ == "__main__":
    main()
