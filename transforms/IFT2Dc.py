# IFT2Dc.py
import numpy as np
from PIL import Image, ImageFile

# 2D CENTERED INVERSE FOURIER TRANSFORM
def IFT2Dc(self, img):
    """
    Applies two-dimensional inverse fourier transform to the image.

    Args:
        img: Input image in fourier domain.

    Returns:
        IFT2D: Input image in spatial domain.
    """
    # 2D array with dimensions equal to the dimensions of leaf image
    Nx, Ny = img.shape
    f1 = np.zeros((Nx, Ny), dtype=complex)
    
    # The exponential of the Inverse Fourier transform
    ix, iy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    f1 = np.exp(1j * np.pi * (ix + iy))
    
    FT = np.fft.ifft2(f1 * img)   # Inverse Fast fourier transform, Amplitude
    IFT2D = f1 * FT               # Exp factor * Amplitude
    
    return IFT2D
