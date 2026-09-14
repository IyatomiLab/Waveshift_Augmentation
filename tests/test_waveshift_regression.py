# test_waveshift_regression.py
# Golden-value regression tests for Wavefront_Shift.
#
# The fingerprints below were captured from the current implementation at
# commit "Fix runnable Waveshift smoke path" (Phase 3a). They encode the
# published-compatible behavior, INCLUDING its known quirks:
#   - propagator grid arange(N) - N//2 - 1 (off by one vs the MATLAB reference)
#   - uint8 conversion by truncation with modular wraparound (no clipping)
#   - z0 drawn per call from random.uniform(1, upper_bound)
# None of these are bugs to fix here. If a refactor changes any number below,
# it changed the science.

import hashlib
import random

import numpy as np
import pytest
from PIL import Image

from conftest import APERTURE_FIXED, IMAGE_SIZE, Z_FIXED
from transforms import FT2Dc, IFT2Dc, PropagatorS
from Waveshift import Wavefront_Shift

# Golden fingerprints: test.JPG -> RGB -> resize(512, 512), z0 = 20.0.
GOLDEN = {
    "s": {
        "shape": (512, 512, 3),
        "dtype": "uint8",
        "min": 0,
        "max": 255,
        "mean": 113.85343805948894,
        "std": 43.50896066510404,
        "sum": 89537987,
        "sha256": "c9659839ac3d3a9d64c69b295d6910ea397583b94438222af2cf895e8b75601a",
    },
    "psf": {
        "shape": (512, 512, 3),
        "dtype": "uint8",
        "min": 8,
        "max": 248,
        "mean": 113.74586868286133,
        "std": 43.04831895130367,
        "sum": 89453391,
        "sha256": "6ad5eef30201ea1775c960bd4d97a9cafab2e3683791c650994da0eef77f689e",
    },
}


def fingerprint(arr):
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sum": int(arr.sum()),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def assert_matches_golden(arr, mode_):
    """Compare against the frozen fingerprint.

    Exact integer fields and sha256 are the real regression guard. mean/std are
    checked with a tight tolerance so that, if sha256 ever fails on a different
    numpy/FFT backend, the statistics say whether the change is float noise
    (statistics match, a pixel or two flipped at a truncation boundary) or a
    genuine behavior change (statistics move).
    """
    got = fingerprint(arr)
    want = GOLDEN[mode_]

    assert got["shape"] == want["shape"]
    assert got["dtype"] == want["dtype"]
    assert got["min"] == want["min"]
    assert got["max"] == want["max"]
    assert got["sum"] == want["sum"]
    assert got["mean"] == pytest.approx(want["mean"], abs=1e-9)
    assert got["std"] == pytest.approx(want["std"], abs=1e-9)
    assert got["sha256"] == want["sha256"]


# --------------------------------------------------------------------------
# 2 & 3. Deterministic WS1 / WS2 output
# --------------------------------------------------------------------------

def test_ws1_deterministic_output(leaf_512, fixed_z):
    """WS 1.0 (mode_='s') at z=20.0 reproduces the frozen fingerprint."""
    out = Wavefront_Shift(mode_="s")(leaf_512)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == IMAGE_SIZE
    assert_matches_golden(np.array(out), "s")


def test_ws2_deterministic_output(leaf_512, fixed_z):
    """WS 2.0 (mode_='psf') at z=20.0, aperture_coeff=0.01."""
    out = Wavefront_Shift(mode_="psf", aperture_coeff=APERTURE_FIXED)(leaf_512)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == IMAGE_SIZE
    assert_matches_golden(np.array(out), "psf")


def test_ws1_and_ws2_differ(leaf_512, fixed_z):
    """The two modes must not collapse to the same result."""
    a = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    b = np.array(Wavefront_Shift(mode_="psf", aperture_coeff=APERTURE_FIXED)(leaf_512))
    assert not np.array_equal(a, b)


def test_repeated_calls_at_fixed_z_are_bitwise_identical(leaf_512, fixed_z):
    a = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    b = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# 4. RGB channel independence
# --------------------------------------------------------------------------

def synthetic_rgb(n=64, green_offset=0):
    """Deterministic RGB image whose three channels differ from each other."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    arr = np.zeros((n, n, 3), dtype=np.int64)
    arr[..., 0] = (xs * 4) % 256
    arr[..., 1] = (ys * 4 + green_offset) % 256
    arr[..., 2] = ((xs + ys) * 2) % 256
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def test_channels_are_processed_independently(fixed_z):
    """Changing only the green channel must leave red and blue untouched.

    This is the observable consequence of splitting the image and propagating
    each channel with its own wavelength.
    """
    a = np.array(Wavefront_Shift(mode_="s")(synthetic_rgb()))
    b = np.array(Wavefront_Shift(mode_="s")(synthetic_rgb(green_offset=37)))

    assert np.array_equal(a[..., 0], b[..., 0]), "red leaked from green"
    assert np.array_equal(a[..., 2], b[..., 2]), "blue leaked from green"
    assert not np.array_equal(a[..., 1], b[..., 1]), "green did not change"


def test_channels_use_distinct_wavelengths(fixed_z):
    """R/G/B are propagated with different lambdas, so outputs differ."""
    out = np.array(Wavefront_Shift(mode_="s")(synthetic_rgb()))
    assert not np.array_equal(out[..., 0], out[..., 1])
    assert not np.array_equal(out[..., 1], out[..., 2])


# --------------------------------------------------------------------------
# 5. Shape / dtype / range behavior
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode_", ["s", "psf"])
def test_square_rgb_in_square_rgb_out(leaf_512, fixed_z, mode_):
    out = Wavefront_Shift(mode_=mode_)(leaf_512)
    assert out.mode == "RGB"
    assert out.size == leaf_512.size
    arr = np.array(out)
    assert arr.dtype == np.uint8
    assert arr.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    assert arr.min() >= 0 and arr.max() <= 255


def test_uint8_wraparound_is_preserved_not_clipped(leaf_512, fixed_z):
    """Magnitudes above 255 wrap modularly; they are NOT clipped to 255.

    This is the published behavior and must survive future cleanup. The
    assertion runs through the public transform, so introducing np.clip inside
    Wavefront_Shift fails this test directly rather than only shifting sha256.
    """
    channel = leaf_512.split()[0]
    propagator = PropagatorS(*leaf_512.size, 620e-9, Z_FIXED)
    magnitude = np.abs(IFT2Dc(FT2Dc(channel) * propagator))

    over = magnitude > 255
    assert over.any(), "expected some magnitudes above 255 at this z"

    wrapped = magnitude[over].astype(np.uint8)
    # Wrapping sends at least one over-range pixel far below 255.
    assert wrapped.min() < 255, "values above 255 should wrap, not saturate"

    # The public transform must reproduce those wrapped values, not clipped ones.
    red = np.array(Wavefront_Shift(mode_="s")(leaf_512))[..., 0]
    assert np.array_equal(red[over], wrapped)
    assert not np.array_equal(red[over], np.full(wrapped.shape, 255, np.uint8))


# --------------------------------------------------------------------------
# 6. Currently known failures (documented, not fixed)
# --------------------------------------------------------------------------

@pytest.mark.xfail(raises=ValueError, strict=True,
                   reason="rectangular input: propagator/FFT shapes transpose")
@pytest.mark.parametrize("mode_", ["s", "psf"])
def test_rectangular_input_currently_fails(fixed_z, mode_):
    leaf = Image.new("RGB", (640, 480), (10, 20, 30))
    Wavefront_Shift(mode_=mode_)(leaf)


@pytest.mark.xfail(raises=ValueError, strict=True,
                   reason="grayscale: leaf.split() yields 1 channel, 3 expected")
def test_grayscale_input_currently_fails(fixed_z):
    leaf = Image.new("L", IMAGE_SIZE, 128)
    Wavefront_Shift(mode_="s")(leaf)


@pytest.mark.xfail(raises=ValueError, strict=True,
                   reason="RGBA: leaf.split() yields 4 channels, 3 expected")
def test_rgba_input_currently_fails(fixed_z):
    leaf = Image.new("RGBA", IMAGE_SIZE, (10, 20, 30, 255))
    Wavefront_Shift(mode_="s")(leaf)


@pytest.mark.xfail(raises=UnboundLocalError, strict=True,
                   reason="unknown mode_: no else branch, propRED never assigned")
def test_invalid_mode_currently_fails(leaf_512, fixed_z):
    Wavefront_Shift(mode_="bogus")(leaf_512)


def test_odd_square_input_currently_works(fixed_z):
    """Documents that odd square sizes are accepted (unlike rectangles)."""
    leaf = Image.new("RGB", (127, 127), (10, 20, 30))
    out = Wavefront_Shift(mode_="s")(leaf)
    assert out.size == (127, 127)


# --------------------------------------------------------------------------
# 7. Reproducibility (current behavior, not yet fixed)
# --------------------------------------------------------------------------

def test_random_seed_controls_the_transform(leaf_512):
    """Python's random module DOES control z0."""
    random.seed(0)
    a = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    random.seed(0)
    b = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    assert np.array_equal(a, b)


def test_numpy_seed_does_not_control_the_transform(leaf_512):
    """np.random.seed does NOT control z0: the transform uses random.uniform.

    This is a known trap for PyTorch/NumPy users. Documented here so that a
    future fix is a deliberate, visible change rather than an accident.
    """
    np.random.seed(0)
    a = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    np.random.seed(0)
    b = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    assert not np.array_equal(a, b)


def test_unseeded_calls_vary(leaf_512):
    """Without a seed, z0 is redrawn per call, so outputs differ."""
    a = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    b = np.array(Wavefront_Shift(mode_="s")(leaf_512))
    assert not np.array_equal(a, b)
