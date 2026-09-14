"""Centered 2-D Fourier transforms.

Two conventions live here:

``modern``
    ``fftshift(fft2(ifftshift(x)))`` -- the textbook centered transform. Correct
    for even and odd sizes and for rectangular arrays.

``legacy``
    The published implementation's checkerboard trick, ``f1 * fft2(f1 * x)``
    with ``f1 = (-1)**(i + j)``. For even-sized arrays this is *identical* to
    the modern convention; for odd sizes it differs by a half-sample shift.
    Kept so that ``compatibility="legacy"`` reproduces the original numbers
    bit-for-bit.

Both are exact inverse pairs and both preserve energy
(``sum|x|^2 == sum|X|^2 / (H*W)``).
"""

import numpy as np

__all__ = ["centered_fft2", "centered_ifft2"]

COMPATIBILITY_MODES = ("modern", "legacy")


def check_compatibility(compatibility):
    """Validate a compatibility mode, raising a helpful ValueError."""
    if compatibility not in COMPATIBILITY_MODES:
        raise ValueError(
            f"compatibility must be one of {COMPATIBILITY_MODES}, "
            f"got {compatibility!r}"
        )
    return compatibility


def _checkerboard(shape, sign):
    """``exp(sign * 1j * pi * (i + j))``, i.e. ``(-1)**(i + j)``."""
    height, width = shape
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.exp(sign * 1j * np.pi * (i + j))


def _as_2d_array(array):
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    return array


def centered_fft2(array, compatibility="modern"):
    """Forward centered 2-D FFT.

    Parameters
    ----------
    array : array_like
        Two-dimensional input, shape ``(height, width)``.
    compatibility : {"modern", "legacy"}
        Transform convention; see the module docstring.
    """
    check_compatibility(compatibility)
    array = _as_2d_array(array)

    if compatibility == "legacy":
        f1 = _checkerboard(array.shape, sign=-1)
        return f1 * np.fft.fft2(f1 * array)

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))


def centered_ifft2(array, compatibility="modern"):
    """Inverse centered 2-D FFT."""
    check_compatibility(compatibility)
    array = _as_2d_array(array)

    if compatibility == "legacy":
        f1 = _checkerboard(array.shape, sign=+1)
        return f1 * np.fft.ifft2(f1 * array)

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array)))
