"""WS2.5-style ROI extension: whole-image vs region-only vs cascaded.

WS2.5 is the PhD thesis extension (Appendix G), NOT part of either journal
paper. Concept adapted from `Local_waveshift.m`, which cropped a square region,
propagated it and pasted it back.

The bottom row zooms into the first ROI: at full-figure scale the change is too
small to see, and a figure that demonstrates nothing is worse than no figure.

Run:  python scripts/generate_assets/generate_roi_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from waveshift import apply_to_regions, waveshift

from _common import Z_DEMO, save, sample_image, show, use_style

BOXES = [(60, 60, 250, 250), (280, 275, 460, 455)]
ZOOM = BOXES[0]

# A deliberately strong aperture, with scale_z="area" so the small region sees
# the same phase excursion a full frame would. At the documented default
# R = 0.01 the ROI effect is invisible at figure scale.
APERTURE_ROI = 0.05


def draw_boxes(ax, color="#e8a33d"):
    for left, top, right, bottom in BOXES:
        ax.add_patch(
            Rectangle((left, top), right - left, bottom - top,
                      fill=False, edgecolor=color, linewidth=1.5)
        )


def roi_pass(image):
    return apply_to_regions(
        image, BOXES, version="ws2", z=Z_DEMO, aperture=APERTURE_ROI,
        feather=12, scale_z="area",
    )


def main():
    use_style()
    image = sample_image()

    roi_only = roi_pass(image)
    # Cascade: a gentle global pass first, then a stronger pass inside the ROIs.
    cascaded = roi_pass(waveshift(image, version="ws1", z=11.0))

    panels = [
        (image, "Original with ROIs"),
        (roi_only, f"ROI only\nWS2 in boxes, z = {Z_DEMO:g}, R = {APERTURE_ROI:g}"),
        (cascaded, "Cascaded\nglobal WS1 (z = 11) + ROI WS2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.4))
    for column, (panel, title) in enumerate(panels):
        show(axes[0, column], panel, title)
        draw_boxes(axes[0, column])
        show(axes[1, column], panel.crop(ZOOM))

    axes[1, 0].set_ylabel("zoom: first ROI", fontsize=9)
    fig.suptitle(
        "WS2.5-style ROI extension — thesis Appendix G, not part of either "
        "journal paper",
        fontsize=11, y=0.965,
    )
    fig.text(0.5, 0.02,
             "Boxes are caller-supplied; no object detector is required. Borders "
             "are feathered, and pixels outside every box are bit-for-bit unchanged.",
             ha="center", fontsize=8.5, color="#666666")
    fig.subplots_adjust(wspace=0.07, hspace=0.04, top=0.88, bottom=0.06)
    save(fig, "ws25_roi_example.png")


if __name__ == "__main__":
    main()
