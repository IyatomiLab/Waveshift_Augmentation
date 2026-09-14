# conftest.py
# Shared fixtures and path setup for the Waveshift regression suite.
#
# These tests FREEZE the current published-compatible behavior. They are a
# safety net for future cleanup: if a refactor changes any number below, the
# change was not behavior-preserving and must be reviewed deliberately.

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# The package lives at the repository root, not inside tests/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fixed hyperparameters used by every regression fingerprint.
# z0 is normally drawn per call by random.uniform(1, upper_bound); the tests
# pin it so the output is deterministic. These values are arbitrary but frozen.
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
    """test.JPG as the README uses it: RGB, resized to 512x512."""
    return Image.open(test_image_path).convert("RGB").resize(IMAGE_SIZE)


@pytest.fixture
def fixed_z(monkeypatch):
    """Pin z0 without touching the public API.

    Wavefront_Shift.__call__ calls random.uniform(1, upper_bound). Patching
    random.uniform is the least invasive way to make the output deterministic
    while leaving the class untouched.
    """
    import random

    monkeypatch.setattr(random, "uniform", lambda a, b: Z_FIXED)
    return Z_FIXED
