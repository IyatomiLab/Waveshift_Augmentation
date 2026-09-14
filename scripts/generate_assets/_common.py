"""Shared helpers for the documentation asset generators.

Every generator in this directory is deterministic: fixed ``z``, fixed
aperture, fixed seeds, no wall-clock or random state. Re-running them
reproduces byte-identical figures.

Paths are always resolved relative to the repository root, so the scripts work
from any working directory and contain no absolute paths.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "docs" / "assets"
SAMPLE = REPO_ROOT / "test.JPG"

#: Figures are rendered at this resolution unless a script overrides it.
#: ~96 dpi keeps every figure around 1000-1200 px wide, which is more than
#: GitHub renders, while keeping the files small.
DPI = 96

#: Parameters shared across figures so the docs tell a consistent story.
SIZE = 512
Z_DEMO = 41.0
APERTURE_DEMO = 0.01

_STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "font.size": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}


def use_style():
    plt.rcParams.update(_STYLE)


def sample_image(size=SIZE):
    """Load the bundled sample as an RGB PIL image.

    Falls back to a deterministic synthetic texture if the sample is missing,
    so the generators never hard-fail on a fresh checkout.
    """
    if SAMPLE.exists():
        return Image.open(SAMPLE).convert("RGB").resize((size, size))
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    base = (
        128
        + 60 * np.sin(xs / 23.0)
        + 40 * np.cos(ys / 17.0)
        + rng.normal(0, 12, (size, size))
    )
    stack = np.dstack([base, base * 0.85 + 20, base * 0.7 + 35])
    return Image.fromarray(np.clip(stack, 0, 255).astype(np.uint8), "RGB")


def log_spectrum(plane):
    """Centered log-magnitude spectrum of a 2-D array, normalized to [0, 1]."""
    from waveshift.fft import centered_fft2

    magnitude = np.log1p(np.abs(centered_fft2(np.asarray(plane, dtype=float))))
    return magnitude / magnitude.max()


def show(ax, data, title=None, cmap=None, vmin=None, vmax=None):
    """Draw an image panel with no ticks."""
    handle = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return handle


def save(fig, name):
    """Save into docs/assets, re-compress, and report the resulting size."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path)
    plt.close(fig)

    # matplotlib's PNG writer leaves a lot on the table; a re-save with
    # optimize=True typically removes 15-25% with no quality change.
    with Image.open(path) as rendered:
        rendered.load()
        rendered.save(path, format="PNG", optimize=True)

    size_kb = path.stat().st_size / 1024
    rel = path.relative_to(REPO_ROOT)
    with Image.open(path) as check:
        width, height = check.size
    print(f"  wrote {rel}  ({size_kb:.0f} KB, {width}x{height})")
    return path
