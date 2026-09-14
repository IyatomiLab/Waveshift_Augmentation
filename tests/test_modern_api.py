"""Tests for the modern WaveShift API.

Covers shape/dtype handling across RGB, grayscale, square and rectangular
inputs; determinism; and the modern-vs-legacy compatibility split.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from conftest import APERTURE_FIXED, Z_FIXED, synthetic_rgb
from waveshift import WaveShift, waveshift
from waveshift.fft import centered_fft2, centered_ifft2
from waveshift.propagators import (
    frequency_grid,
    ws1_propagator,
    ws2_propagator,
)

WS2_KW = {"aperture": APERTURE_FIXED}


# ---------------------------------------------------------------------------
# Shape and dtype handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(64, 64), (80, 48), (48, 80), (47, 31)])
@pytest.mark.parametrize(
    "version,kwargs", [("ws1", {}), ("ws2", WS2_KW)], ids=["ws1", "ws2"]
)
def test_rgb_pil_shapes(size, version, kwargs):
    """RGB PIL input: square, rectangular (both orientations) and odd."""
    image = Image.fromarray(synthetic_rgb(size[1], size[0]), "RGB")
    out = WaveShift(version=version, z=Z_FIXED, **kwargs)(image)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == image.size
    array = np.asarray(out)
    assert array.dtype == np.uint8
    assert array.min() >= 0 and array.max() <= 255


@pytest.mark.parametrize("size", [(64, 64), (80, 48), (48, 80)])
@pytest.mark.parametrize(
    "version,kwargs", [("ws1", {}), ("ws2", WS2_KW)], ids=["ws1", "ws2"]
)
def test_grayscale_pil_shapes(size, version, kwargs):
    """Grayscale PIL input works and stays single-channel."""
    image = Image.fromarray(synthetic_rgb(size[1], size[0]), "RGB").convert("L")
    out = WaveShift(version=version, z=Z_FIXED, **kwargs)(image)
    assert out.mode == "L"
    assert out.size == image.size
    assert np.asarray(out).dtype == np.uint8


@pytest.mark.parametrize(
    "shape", [(48, 64, 3), (48, 64), (48, 64, 1), (64, 64, 3)]
)
def test_numpy_shapes_round_trip(shape):
    """NumPy input returns NumPy of identical shape and dtype."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, shape, dtype=np.uint8)
    out = WaveShift(version="ws1", z=Z_FIXED)(array)
    assert isinstance(out, np.ndarray)
    assert out.shape == array.shape
    assert out.dtype == array.dtype


def test_rgba_alpha_is_preserved_unchanged():
    """Alpha rides through untouched; only the colour planes propagate."""
    rgb = synthetic_rgb(40, 56)
    alpha = np.linspace(0, 255, 40 * 56).reshape(40, 56).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    out = WaveShift(version="ws1", z=Z_FIXED)(rgba)
    assert out.shape == rgba.shape
    assert np.array_equal(out[..., 3], alpha)
    assert not np.array_equal(out[..., :3], rgb)


def test_rgba_pil_alpha_is_preserved():
    rgb = synthetic_rgb(32, 32)
    alpha = np.full((32, 32), 200, np.uint8)
    image = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    out = WaveShift(version="ws1", z=Z_FIXED)(image)
    assert out.mode == "RGBA"
    assert np.array_equal(np.asarray(out)[..., 3], alpha)


def test_float_input_is_not_quantized_or_clipped():
    """Float arrays keep their dtype and may exceed 255: science, not display."""
    array = synthetic_rgb(64, 64).astype(np.float64)
    out = WaveShift(version="ws1", z=Z_FIXED)(array)
    assert out.dtype == np.float64
    assert not np.array_equal(out, np.round(out))  # genuinely continuous


def test_output_coercion():
    array = synthetic_rgb(32, 40)
    as_pil = WaveShift(version="ws1", z=Z_FIXED, output="pil")(array)
    assert isinstance(as_pil, Image.Image)

    image = Image.fromarray(array, "RGB")
    as_numpy = WaveShift(version="ws1", z=Z_FIXED, output="numpy")(image)
    assert isinstance(as_numpy, np.ndarray)
    assert as_numpy.shape == array.shape


# ---------------------------------------------------------------------------
# Determinism and sampling
# ---------------------------------------------------------------------------

def test_fixed_z_is_deterministic():
    image = synthetic_rgb()
    a = WaveShift(version="ws1", z=Z_FIXED)(image)
    b = WaveShift(version="ws1", z=Z_FIXED)(image)
    assert np.array_equal(a, b)


def test_fixed_aperture_is_deterministic():
    image = synthetic_rgb()
    a = WaveShift(version="ws2", z=Z_FIXED, aperture=0.01)(image)
    b = WaveShift(version="ws2", z=Z_FIXED, aperture=0.01)(image)
    assert np.array_equal(a, b)


def test_seeded_z_range_is_reproducible():
    image = synthetic_rgb()
    a = WaveShift(version="ws1", z_range=(1, 41), seed=7)(image)
    b = WaveShift(version="ws1", z_range=(1, 41), seed=7)(image)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_results():
    image = synthetic_rgb()
    a = WaveShift(version="ws1", z_range=(1, 41), seed=7)(image)
    b = WaveShift(version="ws1", z_range=(1, 41), seed=8)(image)
    assert not np.array_equal(a, b)


def test_seeded_aperture_range_is_reproducible():
    image = synthetic_rgb()
    kw = dict(version="ws2", z=Z_FIXED, aperture_range=(0.0001, 0.05), seed=3)
    assert np.array_equal(WaveShift(**kw)(image), WaveShift(**kw)(image))


def test_modern_api_ignores_global_rng_state():
    """Unlike the legacy class, results never depend on global RNG state."""
    import random

    image = synthetic_rgb()
    random.seed(0)
    np.random.seed(0)
    a = WaveShift(version="ws1", z_range=(1, 41), seed=5)(image)
    random.seed(999)
    np.random.seed(999)
    b = WaveShift(version="ws1", z_range=(1, 41), seed=5)(image)
    assert np.array_equal(a, b)


def test_z_range_actually_varies_across_calls():
    image = synthetic_rgb()
    transform = WaveShift(version="ws1", z_range=(1, 41), seed=0)
    assert not np.array_equal(transform(image), transform(image))


def test_degenerate_range_is_treated_as_fixed():
    image = synthetic_rgb()
    a = WaveShift(version="ws1", z_range=(20.0, 20.0))(image)
    b = WaveShift(version="ws1", z=20.0)(image)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# WS1 vs WS2
# ---------------------------------------------------------------------------

def test_ws1_and_ws2_differ():
    image = synthetic_rgb()
    a = WaveShift(version="ws1", z=Z_FIXED)(image)
    b = WaveShift(version="ws2", z=Z_FIXED, aperture=APERTURE_FIXED)(image)
    assert not np.array_equal(a, b)


def test_smaller_aperture_changes_the_image_less():
    """R controls how much high-frequency content the envelope removes."""
    image = synthetic_rgb(seed=1).astype(np.float64)
    reference = image.astype(float)

    def deviation(aperture):
        out = WaveShift(version="ws2", z=Z_FIXED, aperture=aperture)(image)
        return float(np.sqrt(np.mean((out - reference) ** 2)))

    assert deviation(0.001) < deviation(0.05)


def test_ws1_rejects_aperture_arguments():
    with pytest.raises(ValueError, match="WS1 has no aperture"):
        WaveShift(version="ws1", aperture=0.01)
    with pytest.raises(ValueError, match="WS1 has no aperture"):
        WaveShift(version="ws1", aperture_range=(0.001, 0.01))


def test_ws2_defaults_to_published_aperture():
    transform = WaveShift(version="ws2", z=Z_FIXED)
    assert transform.aperture == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_invalid_version_raises_clear_error():
    with pytest.raises(ValueError, match="version must be one of"):
        WaveShift(version="ws3")
    with pytest.raises(ValueError, match="version must be one of"):
        waveshift(synthetic_rgb(), version="ws2.5")


def test_invalid_compatibility_raises():
    with pytest.raises(ValueError, match="compatibility must be one of"):
        WaveShift(version="ws1", compatibility="original")


def test_invalid_output_raises():
    with pytest.raises(ValueError, match="output must be"):
        WaveShift(version="ws1", output="tensor")


def test_z_and_z_range_together_raise():
    with pytest.raises(ValueError, match="either z or z_range"):
        WaveShift(version="ws1", z=5, z_range=(1, 41))(synthetic_rgb())


def test_unsupported_input_type_raises():
    with pytest.raises(TypeError, match="PIL.Image or a NumPy array"):
        WaveShift(version="ws1", z=Z_FIXED)([[1, 2], [3, 4]])


def test_bad_array_shape_raises():
    with pytest.raises(ValueError, match=r"C in \(1, 3, 4\)"):
        WaveShift(version="ws1", z=Z_FIXED)(np.zeros((8, 8, 5), np.uint8))


# ---------------------------------------------------------------------------
# Modern vs legacy compatibility
# ---------------------------------------------------------------------------

LEGACY_GOLDEN = {
    "ws1": "c9659839ac3d3a9d64c69b295d6910ea397583b94438222af2cf895e8b75601a",
    "ws2": "6ad5eef30201ea1775c960bd4d97a9cafab2e3683791c650994da0eef77f689e",
}


@pytest.mark.parametrize("version", ["ws1", "ws2"])
def test_legacy_mode_reproduces_published_fingerprint(leaf_512, version):
    """compatibility='legacy' through the NEW API must match the old numbers.

    This is the bridge test: it proves the rewrite did not drift from the
    published snapshot.
    """
    kwargs = {"aperture": APERTURE_FIXED} if version == "ws2" else {}
    out = waveshift(
        leaf_512, version=version, z=Z_FIXED, compatibility="legacy", **kwargs
    )
    digest = hashlib.sha256(np.asarray(out).tobytes()).hexdigest()
    assert digest == LEGACY_GOLDEN[version]


@pytest.mark.parametrize("version", ["ws1", "ws2"])
def test_legacy_mode_matches_the_original_class(leaf_512, fixed_z, version):
    """The new legacy path and the vendored original class agree bitwise."""
    from waveshift.legacy import Wavefront_Shift

    mode = "s" if version == "ws1" else "psf"
    old = np.asarray(
        Wavefront_Shift(mode_=mode, aperture_coeff=APERTURE_FIXED)(leaf_512)
    )
    kwargs = {"aperture": APERTURE_FIXED} if version == "ws2" else {}
    new = np.asarray(
        waveshift(
            leaf_512, version=version, z=Z_FIXED, compatibility="legacy", **kwargs
        )
    )
    assert np.array_equal(old, new)


def test_modern_and_legacy_differ_on_the_same_input(leaf_512):
    """They must not be silently identical: the grid and clipping differ."""
    modern = np.asarray(waveshift(leaf_512, version="ws1", z=Z_FIXED))
    legacy = np.asarray(
        waveshift(leaf_512, version="ws1", z=Z_FIXED, compatibility="legacy")
    )
    assert not np.array_equal(modern, legacy)
    # ...but they should be close: a one-pixel grid shift, not a new method.
    assert np.abs(modern.astype(int) - legacy.astype(int)).mean() < 1.0


def test_modern_clips_where_legacy_wraps(leaf_512):
    """The headline behavioral difference, asserted directly."""
    raw = WaveShift(version="ws1", z=Z_FIXED, output="numpy")(
        np.asarray(leaf_512, dtype=np.float64)
    )
    over = raw > 255
    assert over.any(), "expected over-range pixels at this z"

    modern = np.asarray(waveshift(leaf_512, version="ws1", z=Z_FIXED))
    legacy = np.asarray(
        waveshift(leaf_512, version="ws1", z=Z_FIXED, compatibility="legacy")
    )
    assert np.all(modern[over] == 255), "modern must saturate"
    assert legacy[over].min() < 255, "legacy must wrap"


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(64, 64), (48, 80), (31, 47)])
@pytest.mark.parametrize("compatibility", ["modern", "legacy"])
def test_centered_transform_round_trip(shape, compatibility):
    rng = np.random.default_rng(0)
    array = rng.normal(size=shape)
    out = centered_ifft2(
        centered_fft2(array, compatibility=compatibility),
        compatibility=compatibility,
    )
    assert np.allclose(out.real, array, atol=1e-9)


@pytest.mark.parametrize("shape", [(64, 64), (48, 80)])
def test_propagator_shapes_follow_height_width(shape):
    """Propagators are built as (height, width): no accidental transpose."""
    assert ws1_propagator(shape, 620e-9, Z_FIXED).shape == shape
    assert ws2_propagator(shape, 620e-9, Z_FIXED, 0.01).shape == shape


def test_modern_grid_is_centered():
    u, v = frequency_grid((8, 8), compatibility="modern")
    assert u.min() == -4 and u.max() == 3
    assert v.min() == -4 and v.max() == 3


def test_legacy_grid_keeps_the_published_offset():
    u, v = frequency_grid((512, 512), compatibility="legacy")
    assert u.min() == -257 and u.max() == 254


def test_ws1_propagator_is_pure_phase():
    p = ws1_propagator((48, 64), 620e-9, Z_FIXED)
    assert np.allclose(np.abs(p), 1.0, atol=1e-12)


def test_ws2_propagator_attenuates():
    q = ws2_propagator((48, 64), 620e-9, Z_FIXED, 0.05)
    magnitude = np.abs(q)
    assert magnitude.max() == pytest.approx(1.0, abs=1e-12)
    assert magnitude.min() < 1.0


def test_ws2_propagator_requires_aperture():
    with pytest.raises(ValueError, match="requires an aperture"):
        ws2_propagator((8, 8), 620e-9, Z_FIXED, None)
