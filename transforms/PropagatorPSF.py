import numpy as np
from scipy.special import j1

def PropagatorPSF(Nx, Ny, lambda_, z0, aperture_coeff):
    """
    Constructs the Airy Disk PSF propagator for a circular aperture.

    Args:
        Nx, Ny: The dimensions of the propagator, same as the input image.
        lambda_: The wavelength of light for the chosen channel.
        z0: The wavefront distance used to construct the propagator shape.
        aperture_coeff: Radius of the circular aperture.

    Returns:
        airy_psf: The normalized Airy Disk PSF in the Fourier domain.
    """
    # Initialize propagator matrices
    airy_psf = np.zeros((Nx, Ny), dtype=complex)

    # Compute spatial frequency coordinates
    u, v = np.meshgrid(np.arange(Nx) - Nx//2 - 1, np.arange(Ny) - Ny//2 - 1, indexing='xy')
    r = np.sqrt(u**2 + v**2)

    # Avoid division by zero
    r[r == 0] = 1e-9  

    # Compute Bessel function J1(r)
    J1r = j1(aperture_coeff * r)

    # Compute Airy disk intensity distribution
    Airy_disk = (2 * J1r / (aperture_coeff * r))**2

    # Compute wavefront propagator
    propagator = np.exp(-1j * np.pi * lambda_ * (u**2 + v**2) * z0)

    # Compute Airy Disk PSF
    airy_psf = Airy_disk * propagator

    # Normalize intensity
    airy_psf /= np.max(np.abs(airy_psf))

    return airy_psf
