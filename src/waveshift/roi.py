"""WS2.5: region-of-interest Waveshift.

Scope
-----
**This is a thesis extension, not part of either journal paper.** WS1 (IEEE
Access 2025) and WS2 (Electronics 2025) operate on the whole image. The ROI
mechanism here follows Appendix G of

    Gent Imeraj, *Machine Learning Across Light and Language: Toward Robust
    and Interpretable Fine-Grained Image Classification*, PhD thesis, 2025.

Cite it as the thesis extension, never as one of the two papers.

Resolution caveat
-----------------
The propagator coordinates are pixel indices, so the strength of a given
``(z, R)`` depends on the size of the array it is applied to. A 100x100 ROI
propagated at ``z=20`` is affected far less than a 512x512 image at the same
``z``. This is a genuine property of the method, not a bug. Two policies are
offered via ``scale_z``:

``"none"`` (default)
    Use ``z`` exactly as given. Honest and predictable; the ROI is simply
    propagated less than a full frame would be.
``"area"``
    Scale ``z`` by ``(full_diagonal / roi_diagonal) ** 2`` so the maximum
    phase excursion across the ROI matches what the full image would see.
    Convenient, but it means the ``z`` you asked for is not the ``z`` applied.

No detector is involved. Boxes come from the caller. An optional YOLO example
is sketched in ``examples/roi_usage.py`` and is never imported here, so
``ultralytics`` is not a dependency.
"""

import numpy as np

from .augmentation import WaveShift

__all__ = ["apply_to_regions", "ROIWaveShift"]

SCALE_POLICIES = ("none", "area")


def _as_array(image):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None
    if Image is not None and isinstance(image, Image.Image):
        return np.asarray(image), True
    return np.asarray(image), False


def _validate_box(box, height, width):
    try:
        left, top, right, bottom = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        raise ValueError(
            f"each box must be (left, top, right, bottom), got {box!r}"
        ) from None

    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 2 or bottom - top < 2:
        raise ValueError(
            f"box {box!r} is empty or smaller than 2x2 after clipping to the "
            f"image bounds ({width}x{height})"
        )
    return left, top, right, bottom


def _scaled_z(z, box, shape, policy):
    if policy == "none":
        return z
    height, width = shape
    left, top, right, bottom = box
    full = float(height**2 + width**2)
    roi = float((bottom - top) ** 2 + (right - left) ** 2)
    if roi == 0:
        return z
    return z * (full / roi)


def _feather_mask(shape, width_px):
    """Smooth 0->1 ramp inset from the border, for seamless compositing."""
    height, width = shape
    if width_px <= 0:
        return np.ones(shape, dtype=float)

    def ramp(n):
        axis = np.ones(n, dtype=float)
        k = min(int(width_px), max(1, n // 2))
        edge = (np.arange(k) + 0.5) / k
        axis[:k] = edge
        axis[n - k:] = edge[::-1]
        return axis

    return np.outer(ramp(height), ramp(width))


def apply_to_regions(
    image,
    boxes,
    version="ws2",
    z=None,
    aperture=None,
    masks=None,
    feather=0,
    scale_z="none",
    compatibility="modern",
    seed=None,
):
    """Apply Waveshift inside each box and composite back into the image.

    Parameters
    ----------
    image : PIL.Image or numpy.ndarray
        Source image. The output has the same type, shape and dtype.
    boxes : sequence of (left, top, right, bottom)
        Pixel coordinates, right/bottom exclusive. Boxes are clipped to the
        image bounds; a box smaller than 2x2 after clipping raises.
    version : {"ws1", "ws2"}
    z, aperture : float, optional
        Passed to :class:`~waveshift.WaveShift`. Give fixed values for
        deterministic output.
    masks : sequence of 2-D arrays, optional
        Per-box soft masks in ``[0, 1]``, shaped like the box. Where a mask is
        0 the original pixels survive untouched.
    feather : int
        Width in pixels of a cosine-free linear ramp applied at box borders to
        avoid hard seams. Combined multiplicatively with ``masks``.
    scale_z : {"none", "area"}
        Resolution policy; see the module docstring.
    compatibility : {"modern", "legacy"}
        ``legacy`` is accepted but rarely sensible here: it only reproduces
        published behavior for square RGB regions.
    seed : int, optional

    Returns
    -------
    Same type as ``image``, with regions replaced.

    Notes
    -----
    Regions are processed in order; overlapping boxes composite cumulatively,
    so a later box sees the result of an earlier one.
    """
    if scale_z not in SCALE_POLICIES:
        raise ValueError(
            f"scale_z must be one of {SCALE_POLICIES}, got {scale_z!r}"
        )

    array, was_pil = _as_array(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"unsupported image shape {array.shape}")
    height, width = array.shape[:2]
    dtype = array.dtype

    boxes = list(boxes)
    if masks is not None:
        masks = list(masks)
        if len(masks) != len(boxes):
            raise ValueError(
                f"got {len(masks)} masks for {len(boxes)} boxes; they must match"
            )

    working = array.astype(float, copy=True)

    for index, raw_box in enumerate(boxes):
        box = _validate_box(raw_box, height, width)
        left, top, right, bottom = box

        region = working[top:bottom, left:right]
        region_z = _scaled_z(z, box, (height, width), scale_z) if z is not None else None

        transform = WaveShift(
            version=version,
            z=region_z,
            aperture=aperture,
            seed=seed,
            compatibility=compatibility,
            output="numpy",
        )
        # Propagate in float. Feeding uint8 here would quantize the region
        # once inside the transform and again after compositing, which shows
        # up as banding at feathered borders.
        propagated = np.asarray(transform(region), dtype=float)

        blend = _feather_mask(region.shape[:2], feather)
        if masks is not None:
            mask = np.asarray(masks[index], dtype=float)
            if mask.shape != region.shape[:2]:
                raise ValueError(
                    f"mask {index} has shape {mask.shape}, expected "
                    f"{region.shape[:2]} to match its box"
                )
            blend = blend * np.clip(mask, 0.0, 1.0)

        if region.ndim == 3:
            blend = blend[..., None]

        working[top:bottom, left:right] = region * (1 - blend) + propagated * blend

    if np.issubdtype(dtype, np.integer):
        # One conversion, at the end, with rounding rather than truncation.
        # Compositing always clips (modern semantics) even when
        # compatibility="legacy" selects the legacy grid and FFT: legacy
        # wraparound inside a blended region would produce black speckles at
        # seams, and ROI work is not part of the published pipeline anyway.
        result = np.clip(np.rint(working), 0, 255).astype(dtype)
    else:
        result = working.astype(dtype)

    if was_pil:
        from PIL import Image

        return Image.fromarray(result, mode=image.mode)
    return result


class ROIWaveShift:
    """Callable wrapper around :func:`apply_to_regions`.

    Boxes may be fixed at construction or supplied per call::

        transform = ROIWaveShift(version="ws2", z=20, aperture=0.01)
        out = transform(image, boxes=[(10, 10, 200, 180)])
    """

    def __init__(
        self,
        version="ws2",
        z=None,
        aperture=None,
        boxes=None,
        feather=0,
        scale_z="none",
        compatibility="modern",
        seed=None,
    ):
        self.version = version
        self.z = z
        self.aperture = aperture
        self.boxes = boxes
        self.feather = feather
        self.scale_z = scale_z
        self.compatibility = compatibility
        self.seed = seed

    def __call__(self, image, boxes=None, masks=None):
        chosen = boxes if boxes is not None else self.boxes
        if chosen is None:
            raise ValueError(
                "no boxes given; pass boxes= to the call or to the constructor"
            )
        return apply_to_regions(
            image,
            chosen,
            version=self.version,
            z=self.z,
            aperture=self.aperture,
            masks=masks,
            feather=self.feather,
            scale_z=self.scale_z,
            compatibility=self.compatibility,
            seed=self.seed,
        )
