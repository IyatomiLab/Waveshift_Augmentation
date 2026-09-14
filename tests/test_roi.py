"""Tests for WS2.5 ROI Waveshift (thesis extension, not a journal paper)."""

import numpy as np
import pytest
from PIL import Image

from conftest import APERTURE_FIXED, Z_FIXED, synthetic_rgb
from waveshift import ROIWaveShift, apply_to_regions

BOX = (10, 8, 50, 40)  # left, top, right, bottom


def textured(height=48, width=64, seed=11):
    """Textured test image.

    ROI tests need real high-frequency content: a smooth gradient changes by
    less than one quantization step at these z values, so an assertion that
    "the region changed" would be testing rounding noise rather than the
    propagator.
    """
    return synthetic_rgb(height, width, seed=seed)


def outside_mask(shape, box):
    left, top, right, bottom = box
    mask = np.ones(shape[:2], dtype=bool)
    mask[top:bottom, left:right] = False
    return mask


def test_shape_and_dtype_are_preserved():
    image = textured(48, 64)
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_pixels_outside_the_box_are_untouched():
    image = textured(48, 64)
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED)
    mask = outside_mask(image.shape, BOX)
    assert np.array_equal(out[mask], image[mask])


def test_pixels_inside_the_box_change():
    image = textured(48, 64)
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED)
    left, top, right, bottom = BOX
    assert not np.array_equal(out[top:bottom, left:right],
                              image[top:bottom, left:right])


def test_works_with_ws2_and_aperture():
    image = textured(48, 64)
    out = apply_to_regions(
        image, [BOX], version="ws2", z=Z_FIXED, aperture=APERTURE_FIXED
    )
    assert out.shape == image.shape
    assert np.array_equal(out[outside_mask(image.shape, BOX)],
                          image[outside_mask(image.shape, BOX)])


def test_grayscale_roi():
    image = textured(48, 64)[..., 0]
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED)
    assert out.shape == image.shape
    assert np.array_equal(out[outside_mask(image.shape, BOX)],
                          image[outside_mask(image.shape, BOX)])


def test_pil_input_returns_pil():
    image = Image.fromarray(textured(48, 64), "RGB")
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED)
    assert isinstance(out, Image.Image)
    assert out.size == image.size
    assert out.mode == "RGB"


def test_multiple_boxes_each_take_effect():
    image = textured(64, 64)
    boxes = [(4, 4, 28, 28), (34, 34, 60, 60)]
    out = apply_to_regions(image, boxes, version="ws1", z=Z_FIXED)
    for left, top, right, bottom in boxes:
        assert not np.array_equal(out[top:bottom, left:right],
                                  image[top:bottom, left:right])
    untouched = np.ones(image.shape[:2], bool)
    for left, top, right, bottom in boxes:
        untouched[top:bottom, left:right] = False
    assert np.array_equal(out[untouched], image[untouched])


def test_zero_mask_leaves_the_region_unchanged():
    """A mask of zeros means 'composite nothing', so the image is untouched."""
    image = textured(48, 64)
    left, top, right, bottom = BOX
    mask = np.zeros((bottom - top, right - left), dtype=float)
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED, masks=[mask])
    assert np.array_equal(out, image)


def test_partial_mask_blends():
    image = textured(48, 64)
    left, top, right, bottom = BOX
    mask = np.zeros((bottom - top, right - left), dtype=float)
    mask[: (bottom - top) // 2] = 1.0
    out = apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED, masks=[mask])
    half = top + (bottom - top) // 2
    assert not np.array_equal(out[top:half, left:right], image[top:half, left:right])
    assert np.array_equal(out[half:bottom, left:right], image[half:bottom, left:right])


def test_feather_keeps_borders_closer_to_the_original():
    # Textured input: the smooth gradient barely responds at this z, so the
    # comparison would sit in quantization noise.
    image = textured(64, 64)
    box = (8, 8, 56, 56)
    hard = apply_to_regions(image, [box], version="ws1", z=Z_FIXED, feather=0)
    soft = apply_to_regions(image, [box], version="ws1", z=Z_FIXED, feather=6)

    def border_delta(result):
        return np.abs(result[8, 8:56].astype(int) - image[8, 8:56].astype(int)).mean()

    assert border_delta(soft) < border_delta(hard)


def test_deterministic_with_fixed_parameters():
    image = textured(48, 64)
    a = apply_to_regions(image, [BOX], version="ws2", z=Z_FIXED, aperture=0.01)
    b = apply_to_regions(image, [BOX], version="ws2", z=Z_FIXED, aperture=0.01)
    assert np.array_equal(a, b)


def test_scale_z_area_increases_the_effect_on_a_small_roi():
    """Documents the resolution caveat: the same z does less on a small ROI.

    Uses textured input, because a smooth gradient has almost no high-frequency
    content for the propagator to act on.
    """
    image = textured(64, 64)
    box = (16, 16, 48, 48)
    plain = apply_to_regions(image, [box], version="ws1", z=Z_FIXED, scale_z="none")
    scaled = apply_to_regions(image, [box], version="ws1", z=Z_FIXED, scale_z="area")

    def delta(result):
        left, top, right, bottom = box
        return np.abs(
            result[top:bottom, left:right].astype(int)
            - image[top:bottom, left:right].astype(int)
        ).mean()

    assert delta(scaled) > delta(plain)


def test_roi_class_accepts_boxes_at_construction_or_call():
    image = textured(48, 64)
    preset = ROIWaveShift(version="ws1", z=Z_FIXED, boxes=[BOX])
    per_call = ROIWaveShift(version="ws1", z=Z_FIXED)
    assert np.array_equal(preset(image), per_call(image, boxes=[BOX]))


def test_roi_class_without_boxes_raises():
    with pytest.raises(ValueError, match="no boxes given"):
        ROIWaveShift(version="ws1", z=Z_FIXED)(textured())


def test_box_is_clipped_to_image_bounds():
    image = textured(32, 32)
    out = apply_to_regions(image, [(-10, -10, 1000, 1000)], version="ws1", z=Z_FIXED)
    assert out.shape == image.shape


def test_degenerate_box_raises():
    image = textured(32, 32)
    with pytest.raises(ValueError, match="empty or smaller than 2x2"):
        apply_to_regions(image, [(5, 5, 6, 6)], version="ws1", z=Z_FIXED)


def test_mask_count_must_match_boxes():
    image = textured(48, 64)
    with pytest.raises(ValueError, match="must match"):
        apply_to_regions(image, [BOX], version="ws1", z=Z_FIXED, masks=[])


def test_mask_shape_must_match_its_box():
    image = textured(48, 64)
    with pytest.raises(ValueError, match="expected"):
        apply_to_regions(
            image, [BOX], version="ws1", z=Z_FIXED, masks=[np.ones((3, 3))]
        )


def test_invalid_scale_policy_raises():
    with pytest.raises(ValueError, match="scale_z must be one of"):
        apply_to_regions(
            textured(), [BOX], version="ws1", z=Z_FIXED, scale_z="magic"
        )
