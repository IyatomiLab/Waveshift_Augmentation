"""Fourier-domain propagators for Waveshift.

WS1 (``ws1_propagator``) is the paraxial Fresnel transfer function for
spherical waves::

    p(u, v) = exp(-i * pi * wavelength * z * (u^2 + v^2))

WS2 (``ws2_propagator``) multiplies that phase by an Airy-disk intensity
envelope for a circular aperture of coefficient ``R``::

    A(u, v) = (2 * J1(R * r) / (R * r))^2,    r = sqrt(u^2 + v^2)
    q(u, v) = A(u, v) * p(u, v)

Units
-----
``u`` and ``v`` are **integer pixel indices**, not calibrated spatial
frequencies: there is no pixel pitch ``dx`` in the expression. The physical
Fresnel kernel would use ``f = u / (N * dx)``. Consequently ``wavelength * z``
is a *lumped, empirical* parameter and ``R`` has units of 1/pixel. A given
``(z, R)`` therefore produces a different effect at a different image
resolution -- see ``docs/theory.md``.

Coordinate grid
---------------
``modern``
    ``arange(N) - N // 2``  ->  ``[-N/2, N/2 - 1]``, zero on the DC bin.
``legacy``
    ``arange(N) - N // 2 - 1``  ->  ``[-N/2 - 1, N/2 - 2]``.
    One pixel off-centre. This came from transliterating 1-based MATLAB
    indices to 0-based NumPy and is preserved because the published results
    were generated with it.
"""

import numpy as np
from scipy.special import j1

from .fft import check_compatibility

__all__ = ["ws1_propagator", "ws2_propagator", "frequency_grid"]

#: Per-channel wavelengths (metres) used by the papers.
WAVELENGTHS_RGB = (620e-9, 535e-9, 450e-9)

#: Wavelength used when an image has a single channel.
WAVELENGTH_GRAY = 535e-9


def _check_shape(shape):
    try:
        height, width = shape
    except (TypeError, ValueError):
        raise ValueError(
            f"shape must be a (height, width) pair, got {shape!r}"
        ) from None
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError(f"shape must be positive, got {(height, width)}")
    return height, width


def frequency_grid(shape, compatibility="modern"):
    """Return the ``(u, v)`` coordinate grids for ``shape = (height, width)``.

    Both returned arrays have shape ``(height, width)``. ``u`` varies along the
    width (columns), ``v`` along the height (rows). Rectangular shapes are
    handled naturally; square is just a special case.
    """
    check_compatibility(compatibility)
    height, width = _check_shape(shape)

    offset = 1 if compatibility == "legacy" else 0
    v_axis = np.arange(height) - height // 2 - offset
    u_axis = np.arange(width) - width // 2 - offset

    # indexing="ij" keeps the first axis as rows, so no accidental transpose.
    v, u = np.meshgrid(v_axis, u_axis, indexing="ij")
    return u, v


def ws1_propagator(shape, wavelength, z, compatibility="modern"):
    """WS1 propagator: pure phase, unit modulus everywhere.

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` of the image being propagated.
    wavelength : float
        Wavelength in metres.
    z : float
        Propagation distance (model units; see module docstring).
    compatibility : {"modern", "legacy"}
        Selects the coordinate grid.
    """
    u, v = frequency_grid(shape, compatibility=compatibility)
    return np.exp(-1j * np.pi * wavelength * (u**2 + v**2) * z)


def ws2_propagator(shape, wavelength, z, aperture, compatibility="modern"):
    """WS2 propagator: WS1 phase modulated by an Airy-disk intensity envelope.

    Unlike WS1 this is *not* unit modulus -- the envelope attenuates high
    spatial frequencies, so WS2 is amplitude-modulating as well as
    phase-modulating.

    Parameters
    ----------
    aperture : float
        Aperture coefficient ``R`` (1/pixel). Larger values shrink the Airy
        main lobe and suppress more high-frequency content.
    """
    if aperture is None:
        raise ValueError("ws2_propagator requires an aperture value")
    aperture = float(aperture)
    if aperture == 0:
        raise ValueError("aperture must be non-zero")

    u, v = frequency_grid(shape, compatibility=compatibility)
    r = np.sqrt(u**2 + v**2)

    # (2*J1(x)/x)^2 -> 1 as x -> 0; fill the removable singularity directly.
    scaled = aperture * r
    airy = np.ones_like(scaled, dtype=float)
    nonzero = scaled != 0
    airy[nonzero] = (2 * j1(scaled[nonzero]) / scaled[nonzero]) ** 2

    phase = np.exp(-1j * np.pi * wavelength * (u**2 + v**2) * z)
    propagator = airy * phase

    # The Airy peak is exactly 1, so this is a no-op in practice. It is kept
    # because the published implementation normalized here.
    peak = np.max(np.abs(propagator))
    if peak > 0:
        propagator = propagator / peak
    return propagator
