"""Public Waveshift API: :class:`WaveShift` and :func:`waveshift`.

Compatibility modes
-------------------
``modern`` (default)
    Centered coordinate grid, safe clipping to ``[0, 255]`` before the uint8
    cast, works on grayscale, RGB, rectangular and square images.

``legacy``
    Reproduces the published GitHub snapshot: off-centre coordinate grid, the
    checkerboard FFT convention, and uint8 **wraparound** instead of clipping.
    Exact reproduction is guaranteed for square RGB uint8 input, which is what
    the papers used.

Randomness
----------
The modern API never touches global RNG state. ``seed`` creates a private
``numpy.random.Generator``, so results are unaffected by ``random.seed``,
``np.random.seed`` or ``torch.manual_seed``. Pass ``seed`` for reproducibility;
pass fixed ``z`` (and ``aperture``) for a fully deterministic transform.

Under a PyTorch ``DataLoader`` with multiple workers, give each worker its own
``WaveShift(seed=...)`` (or leave ``seed=None`` for independent entropy per
worker) rather than relying on ``torch.manual_seed``.
"""

import numpy as np

from .fft import centered_fft2, centered_ifft2, check_compatibility
from .propagators import (
    WAVELENGTH_GRAY,
    WAVELENGTHS_RGB,
    ws1_propagator,
    ws2_propagator,
)

__all__ = ["WaveShift", "waveshift"]

VERSIONS = ("ws1", "ws2")

#: Default aperture coefficient for WS2, matching the published default.
DEFAULT_APERTURE = 0.01

try:  # Pillow is a hard dependency, but keep the import defensive and local.
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    Image = None
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image plumbing
# ---------------------------------------------------------------------------

def _is_pil(image):
    return _PIL_AVAILABLE and isinstance(image, Image.Image)


def _decompose(image):
    """Normalize any supported input to ``(planes, rebuild)``.

    ``planes`` is a list of 2-D float arrays to propagate. ``rebuild`` takes a
    list of processed 2-D float arrays and returns an object matching the
    original input's type, shape and dtype.

    Supported: PIL images (L, RGB, RGBA and anything convertible) and NumPy
    arrays that are 2-D, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``.
    An RGBA alpha channel is carried through unchanged.
    """
    if _is_pil(image):
        return _decompose_pil(image)
    if isinstance(image, np.ndarray):
        return _decompose_array(image)
    raise TypeError(
        "image must be a PIL.Image or a NumPy array, got "
        f"{type(image).__name__}"
    )


def _decompose_pil(image):
    mode = image.mode
    if mode not in ("L", "RGB", "RGBA"):
        # Palette, CMYK, 16-bit and friends: convert to the nearest supported
        # mode rather than failing, and say so in the returned mode.
        mode = "RGBA" if "A" in image.getbands() else "RGB"
        image = image.convert(mode)

    array = np.asarray(image)
    if mode == "L":
        planes = [array.astype(float)]
        alpha = None
    else:
        planes = [array[..., i].astype(float) for i in range(3)]
        alpha = array[..., 3] if mode == "RGBA" else None

    def rebuild(processed, compatibility):
        stacked = _to_uint8_stack(processed, compatibility)
        if mode == "L":
            return Image.fromarray(stacked[..., 0], mode="L")
        if alpha is None:
            return Image.fromarray(stacked, mode="RGB")
        return Image.fromarray(
            np.dstack([stacked, alpha]).astype(np.uint8), mode="RGBA"
        )

    return planes, rebuild


def _decompose_array(array):
    if array.ndim == 2:
        channels, alpha = 1, None
    elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
        channels = min(array.shape[2], 3)
        alpha = array[..., 3] if array.shape[2] == 4 else None
    else:
        raise ValueError(
            "array image must be 2-D or (H, W, C) with C in (1, 3, 4), got "
            f"shape {array.shape}"
        )

    dtype = array.dtype
    if array.ndim == 2:
        planes = [array.astype(float)]
    else:
        planes = [array[..., i].astype(float) for i in range(channels)]

    def rebuild(processed, compatibility):
        if dtype == np.uint8:
            stacked = _to_uint8_stack(processed, compatibility)
        else:
            # Anything else (float, uint16, ...) keeps its dtype with no
            # quantization and no clipping in either compatibility mode.
            # Callers working outside 8-bit are doing science, not producing
            # display images.
            stacked = np.dstack(processed).astype(dtype)

        if array.ndim == 2:
            return stacked[..., 0]
        if alpha is not None:
            return np.dstack([stacked, alpha.astype(dtype)])
        if array.shape[2] == 1:
            return stacked[..., :1]
        return stacked

    return planes, rebuild


def _to_uint8_stack(processed, compatibility):
    """Cast propagated magnitudes to uint8.

    ``legacy``
        Bare ``.astype(np.uint8)``: truncates *and wraps*, so 259.4 becomes 3.
        This loses occasional highlight pixels, but it is what produced the
        published results, so it is preserved exactly.

    ``modern``
        Clip to ``[0, 255]`` first, so an over-range highlight saturates to
        white instead of wrapping to black.
    """
    if compatibility == "modern":
        processed = [np.clip(p, 0.0, 255.0) for p in processed]
    return np.dstack([p.astype(np.uint8) for p in processed])


def _wavelengths_for(n_planes):
    if n_planes == 1:
        return (WAVELENGTH_GRAY,)
    return WAVELENGTHS_RGB[:n_planes]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _propagate_plane(plane, wavelength, version, z, aperture, compatibility):
    shape = plane.shape
    if version == "ws1":
        propagator = ws1_propagator(
            shape, wavelength, z, compatibility=compatibility
        )
    else:
        propagator = ws2_propagator(
            shape, wavelength, z, aperture, compatibility=compatibility
        )

    spectrum = centered_fft2(plane, compatibility=compatibility)
    # The raw magnitude is returned unmodified; clipping vs wraparound is
    # decided at the uint8 cast, so float inputs stay untouched.
    return np.abs(
        centered_ifft2(spectrum * propagator, compatibility=compatibility)
    )


def _validate_version(version):
    if version not in VERSIONS:
        raise ValueError(
            f"version must be one of {VERSIONS}, got {version!r}. "
            "Use 'ws1' for the IEEE Access method or 'ws2' for the "
            "aperture-controlled Electronics method."
        )
    return version


def _resolve_range(value, value_range, name, rng):
    """Pick a scalar from ``value`` or sample it from ``value_range``."""
    if value is not None and value_range is not None:
        raise ValueError(f"pass either {name} or {name}_range, not both")
    if value is not None:
        return float(value)
    if value_range is None:
        return None
    try:
        low, high = value_range
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}_range must be a (low, high) pair, got {value_range!r}"
        ) from None
    low, high = float(low), float(high)
    if low > high:
        raise ValueError(f"{name}_range low must not exceed high, got {(low, high)}")
    if low == high:
        return low
    return float(rng.uniform(low, high))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WaveShift:
    """Physics-driven Waveshift augmentation.

    Parameters
    ----------
    version : {"ws1", "ws2"}
        ``ws1`` applies phase-only Fresnel propagation (IEEE Access 2025).
        ``ws2`` additionally applies an Airy-disk aperture envelope
        (Electronics 2025).
    z, z_range : float or (float, float), optional
        Propagation distance. Give ``z`` for a fixed value or ``z_range`` to
        sample uniformly per call. Defaults to ``z_range=(1, 41)``.
    aperture, aperture_range : float or (float, float), optional
        WS2 aperture coefficient ``R``. Ignored by WS1, which raises if either
        is set, so that a mistaken WS1 call does not silently drop the
        aperture. Defaults to ``aperture=0.01`` for WS2.
    seed : int, optional
        Seed for the private RNG used to sample ``z`` and ``aperture``. Global
        RNG state is never read or written.
    compatibility : {"modern", "legacy"}
        ``modern`` clips before the uint8 cast and uses a centered grid.
        ``legacy`` reproduces the published snapshot exactly.
    output : {"same", "pil", "numpy"}
        ``same`` returns the input type. ``pil`` and ``numpy`` force a type.

    Examples
    --------
    >>> transform = WaveShift(version="ws1", z_range=(1, 41), seed=0)
    >>> augmented = transform(image)                      # doctest: +SKIP
    """

    def __init__(
        self,
        version="ws1",
        z=None,
        z_range=None,
        aperture=None,
        aperture_range=None,
        seed=None,
        compatibility="modern",
        output="same",
    ):
        self.version = _validate_version(version)
        check_compatibility(compatibility)
        if output not in ("same", "pil", "numpy"):
            raise ValueError(
                f"output must be 'same', 'pil' or 'numpy', got {output!r}"
            )

        if version == "ws1" and (aperture is not None or aperture_range is not None):
            raise ValueError(
                "WS1 has no aperture parameter; use version='ws2' to control "
                "the aperture, or drop the aperture argument."
            )
        if version == "ws2" and aperture is None and aperture_range is None:
            aperture = DEFAULT_APERTURE
        if z is None and z_range is None:
            z_range = (1.0, 41.0)

        self.z = z
        self.z_range = z_range
        self.aperture = aperture
        self.aperture_range = aperture_range
        self.compatibility = compatibility
        self.output = output
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        parts = [f"version={self.version!r}"]
        parts.append(f"z={self.z!r}" if self.z is not None else f"z_range={self.z_range!r}")
        if self.version == "ws2":
            if self.aperture is not None:
                parts.append(f"aperture={self.aperture!r}")
            else:
                parts.append(f"aperture_range={self.aperture_range!r}")
        parts.append(f"compatibility={self.compatibility!r}")
        return f"WaveShift({', '.join(parts)})"

    def sample_parameters(self):
        """Draw the ``(z, aperture)`` pair this call will use."""
        z = _resolve_range(self.z, self.z_range, "z", self._rng)
        if self.version == "ws1":
            return z, None
        aperture = _resolve_range(
            self.aperture, self.aperture_range, "aperture", self._rng
        )
        return z, aperture

    def __call__(self, image):
        z, aperture = self.sample_parameters()
        planes, rebuild = _decompose(image)
        wavelengths = _wavelengths_for(len(planes))

        processed = [
            _propagate_plane(
                plane, wavelength, self.version, z, aperture, self.compatibility
            )
            for plane, wavelength in zip(planes, wavelengths)
        ]

        result = rebuild(processed, self.compatibility)
        return _coerce_output(result, self.output, image)


def _coerce_output(result, output, original):
    if output == "same":
        return result
    if output == "numpy":
        return np.asarray(result)
    if not _PIL_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Pillow is required for output='pil'")
    if _is_pil(result):
        return result
    array = np.asarray(result)
    if array.dtype != np.uint8:
        raise ValueError(
            "output='pil' requires uint8 data; got dtype "
            f"{array.dtype}. Convert the input to uint8 first."
        )
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.shape[2] == 1:
        return Image.fromarray(array[..., 0], mode="L")
    return Image.fromarray(array, mode="RGB" if array.shape[2] == 3 else "RGBA")


def waveshift(
    image,
    version="ws1",
    z=None,
    aperture=None,
    compatibility="modern",
    output="same",
    seed=None,
):
    """Apply Waveshift once, functionally.

    A thin wrapper over :class:`WaveShift` for deterministic one-shot use.

    Examples
    --------
    >>> out = waveshift(image, version="ws2", z=41, aperture=0.01)  # doctest: +SKIP
    """
    transform = WaveShift(
        version=version,
        z=z,
        aperture=aperture,
        seed=seed,
        compatibility=compatibility,
        output=output,
    )
    return transform(image)
