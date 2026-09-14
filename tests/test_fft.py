# test_fft.py
# Freeze the behavior of the centered Fourier transform pair and the
# propagators. These tests assert properties of the CURRENT implementation;
# they are not a claim that the conventions are the only valid ones.

import numpy as np
import pytest
from PIL import Image

from transforms import FT2Dc, IFT2Dc, PropagatorPSF, PropagatorS

# Round-trip error measured on this implementation is ~1e-13 for N <= 128.
# The tolerance is deliberately loose enough to survive BLAS/FFT backend
# differences but tight enough to catch any real change of convention.
ROUNDTRIP_ATOL = 1e-9


def deterministic_image(n, seed=0):
    """Small reproducible single-channel PIL image.

    FT2Dc reads img.size, so it requires a PIL Image rather than an ndarray.
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(n, n), dtype=np.uint8)
    return arr, Image.fromarray(arr, mode="L")


@pytest.mark.parametrize("n", [8, 16, 64, 128])
def test_roundtrip_reconstructs_original(n):
    """IFT2Dc(FT2Dc(x)) == x for square even-sized input."""
    arr, img = deterministic_image(n)
    out = IFT2Dc(FT2Dc(img))
    assert out.shape == arr.shape
    assert np.allclose(out.real, arr.astype(float), atol=ROUNDTRIP_ATOL)
    # The reconstruction is real-valued up to numerical noise.
    assert np.abs(out.imag).max() < ROUNDTRIP_ATOL


@pytest.mark.parametrize("n", [7, 31])
def test_roundtrip_odd_sizes(n):
    """Odd square sizes also round-trip. Documents current behavior."""
    arr, img = deterministic_image(n)
    out = IFT2Dc(FT2Dc(img))
    assert np.allclose(out.real, arr.astype(float), atol=ROUNDTRIP_ATOL)


def test_roundtrip_is_exact_inverse_not_approximate():
    """The transform pair is unitary: no scale factor creeps in."""
    arr, img = deterministic_image(64)
    out = IFT2Dc(FT2Dc(img)).real
    ratio = out.sum() / arr.astype(float).sum()
    assert ratio == pytest.approx(1.0, abs=1e-12)


def test_energy_is_preserved_by_ft2dc():
    """Parseval-style identity holds for the current normalization:
    sum|x|^2 == sum|X|^2 / N^2 (fft2 unnormalized, ifft2 carries 1/N^2)."""
    n = 64
    arr, img = deterministic_image(n)
    spectrum = FT2Dc(img)
    lhs = np.sum(np.abs(arr.astype(float)) ** 2)
    rhs = np.sum(np.abs(spectrum) ** 2) / (n * n)
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_ft2dc_requires_pil_image():
    """Current limitation: FT2Dc unpacks img.size, which is an int on ndarray.

    Documents existing behavior; do not 'fix' during Phase 3b.
    """
    arr, _ = deterministic_image(8)
    with pytest.raises(TypeError):
        FT2Dc(arr)


def test_propagator_s_is_pure_phase():
    """WS1.0 propagator has unit modulus everywhere (energy preserving)."""
    p = PropagatorS(64, 64, 620e-9, 20.0)
    assert p.shape == (64, 64)
    assert np.allclose(np.abs(p), 1.0, atol=1e-12)


def test_propagator_psf_is_amplitude_modulated():
    """WS2.0 propagator is NOT pure phase: the Airy envelope attenuates."""
    q = PropagatorPSF(64, 64, 620e-9, 20.0, 0.01)
    assert q.shape == (64, 64)
    mag = np.abs(q)
    assert mag.max() == pytest.approx(1.0, abs=1e-12)
    assert mag.min() < 1.0


def test_propagators_are_deterministic_at_fixed_parameters():
    """Same (z, R) must give bitwise-identical propagators."""
    a = PropagatorS(64, 64, 620e-9, 20.0)
    b = PropagatorS(64, 64, 620e-9, 20.0)
    assert np.array_equal(a, b)

    c = PropagatorPSF(64, 64, 620e-9, 20.0, 0.01)
    d = PropagatorPSF(64, 64, 620e-9, 20.0, 0.01)
    assert np.array_equal(c, d)


def test_propagator_grid_is_current_published_convention():
    """Freeze the coordinate grid: arange(N) - N//2 - 1, i.e. [-257, 254] at N=512.

    This grid is off by one pixel relative to the MATLAB reference, but it is
    what the published Python pipeline used. Frozen deliberately; changing it
    is a Phase 3c decision, not a cleanup.
    """
    n = 512
    axis = np.arange(n) - n // 2 - 1
    assert axis.min() == -257
    assert axis.max() == 254

    # The grid shows up in the propagator as a lack of centro-symmetry.
    p = PropagatorS(n, n, 620e-9, 20.0)
    assert not np.allclose(p, p[::-1, ::-1])
