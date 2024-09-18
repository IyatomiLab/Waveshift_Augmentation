# CCWind.py
import numpy as np
from PIL import Image, ImageFile

# CENTER CROP WINDOW OF AN IMAGE
def CCWind(self, img, size):
    """
    A window to center-crop a given image.

    Args:
        img: Input image.
        size: The side lenth of the center-squared image.

    Returns:
        img.crop((left, top, right, bottom)): The center-squared image.
    """
    width, height = img.size
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2
    return img.crop((left, top, right, bottom))
