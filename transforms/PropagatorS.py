# PropagatorS.py
import numpy as np
from PIL import Image, ImageFile

# WAVEFRONT PROPAGATION FOR SPHERICAL WAVES IN PARAXIAL APPROXIMATION
def PropagatorS(self, Nx, Ny, lambda_, z0):
    # 2D array with dimensions equal to the dimensions of leaf image
    p = np.zeros((Nx, Ny), dtype=complex)
    
    # Propagator [See Chapter 1: WAVE PROPAGATION AND DIFFRACTION THEORY]
    u, v = np.meshgrid(np.arange(Nx) - Nx//2 - 1, np.arange(Ny) - Ny//2 - 1, indexing='xy')
    p = np.exp(-1j * np.pi * lambda_ * (u**2 + v**2) * z0)

    return p
