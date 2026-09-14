# conftest.py
# Shared fixtures for the Waveshift test suite.
#
# The legacy tests FREEZE the published behavior: they are a safety net for
# future cleanup. If a refactor changes any fingerprint in
# test_legacy_regression.py, the change was not behavior-preserving and must
# be reviewed deliberately.
#
# The package is expected to be installed (`pip install -e .`), so there is no
# sys.path manipulation here.

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed hyperparameters used by every regression fingerprint.
Z_FIXED = 20.0
APERTURE_FIXED = 0.01
IMAGE_SIZE = (512, 512)


@pytest.fixture(scope="session")
def test_image_path():
    path = REPO_ROOT / "test.JPG"
    if not path.exists():
        pytest.skip(f"sample image not found: {path}")
    return path


@pytest.fixture(scope="session")
def leaf_512(test_image_path):
    """test.JPG as the papers used it: RGB, resized to 512x512."""
    return Image.open(test_image_path).convert("RGB").resize(IMAGE_SIZE)


@pytest.fixture(scope="session")
def leaf_rect(test_image_path):
    """A deliberately non-square version, 640x480."""
    return Image.open(test_image_path).convert("RGB").resize((640, 480))


@pytest.fixture
def fixed_z(monkeypatch):
    """Pin z0 for the LEGACY class without touching its public API.

    waveshift.legacy.Wavefront_Shift.__call__ calls random.uniform(1,
    upper_bound). Patching random.uniform is the least invasive way to make
    its output deterministic.

    The modern API does not need this: pass z=... or seed=... instead.
    """
    import random

    monkeypatch.setattr(random, "uniform", lambda a, b: Z_FIXED)
    return Z_FIXED


def synthetic_rgb(height=48, width=64, offset=0, seed=None):
    """Deterministic RGB array whose three channels differ from each other."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    arr = np.zeros((height, width, 3), dtype=np.int64)
    arr[..., 0] = (xs * 4) % 256
    arr[..., 1] = (ys * 4 + offset) % 256
    arr[..., 2] = ((xs + ys) * 2) % 256
    return arr.astype(np.uint8)
