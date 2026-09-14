"""Waveshift: physics-driven data augmentation by wavefront propagation.

Quick start
-----------
>>> from waveshift import WaveShift
>>> transform = WaveShift(version="ws1", z_range=(1, 41))
>>> out = transform(image)                                    # doctest: +SKIP

>>> transform = WaveShift(version="ws2", z_range=(15, 151),
...                       aperture_range=(0.0001, 0.01))
>>> out = transform(image)                                    # doctest: +SKIP

Deterministic, one shot:

>>> from waveshift import waveshift
>>> out = waveshift(image, version="ws2", z=41, aperture=0.01)  # doctest: +SKIP

Versions
--------
``ws1``
    Imeraj & Iyatomi, *IEEE Access* **13**, 31303-31317 (2025).
``ws2``
    Imeraj & Iyatomi, *Electronics* **14**, 1735 (2025).
``ws2.5`` (:mod:`waveshift.roi`)
    PhD thesis extension, Appendix G. Not part of either paper.

Reproducing the published snapshot
----------------------------------
Pass ``compatibility="legacy"``, or use :mod:`waveshift.legacy` directly for
the original class verbatim.
"""

from .augmentation import DEFAULT_APERTURE, VERSIONS, WaveShift, waveshift
from .fft import centered_fft2, centered_ifft2
from .propagators import (
    WAVELENGTH_GRAY,
    WAVELENGTHS_RGB,
    frequency_grid,
    ws1_propagator,
    ws2_propagator,
)
from .roi import ROIWaveShift, apply_to_regions

__version__ = "0.3.0"

__all__ = [
    "WaveShift",
    "waveshift",
    "ROIWaveShift",
    "apply_to_regions",
    "centered_fft2",
    "centered_ifft2",
    "ws1_propagator",
    "ws2_propagator",
    "frequency_grid",
    "WAVELENGTHS_RGB",
    "WAVELENGTH_GRAY",
    "DEFAULT_APERTURE",
    "VERSIONS",
    "__version__",
]
