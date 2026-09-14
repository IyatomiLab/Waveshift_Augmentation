"""The original published implementation, preserved verbatim.

This module is the GitHub snapshot of the Waveshift code as used for

* Imeraj & Iyatomi, *IEEE Access* **13**, 31303-31317 (2025)  -- WS1
* Imeraj & Iyatomi, *Electronics* **14**, 1735 (2025)          -- WS2

It is kept **byte-for-byte behavior-compatible** so published results stay
reproducible. Its quirks are intentional and must not be "fixed" here:

* propagator grid ``arange(N) - N//2 - 1`` (one pixel off-centre; an artefact
  of transliterating 1-based MATLAB indices to 0-based NumPy)
* ``.astype(np.uint8)`` truncation with **modular wraparound**, not clipping
* ``FT2Dc`` reads ``img.size`` (width, height) while the array is
  (height, width), so only square images work
* RGB-only: ``leaf.split()`` assumes exactly three channels
* ``z0`` drawn from Python's global ``random`` module, so ``np.random.seed``
  and ``torch.manual_seed`` do **not** control it

For new work use :class:`waveshift.WaveShift` instead. To reproduce these
numbers through the modern API, pass ``compatibility="legacy"``.

The two remaining Phase-3a fixes are included (they only made the code
runnable and changed no numbers): ``self.mode_`` is read correctly and the
propagators are called as module-level functions.
"""

import random

import numpy as np
from PIL import Image
from scipy.special import j1

__all__ = [
    "FT2Dc",
    "IFT2Dc",
    "PropagatorS",
    "PropagatorPSF",
    "CCWind",
    "Wavefront_Shift",
]


def FT2Dc(img):
    """2-D centered Fourier transform (original implementation).

    Note ``Nx, Ny = img.size`` reads (width, height) from a PIL image while
    ``np.asarray(img)`` is (height, width). Square input only.
    """
    Nx, Ny = img.size
    ix, iy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    f1 = np.exp(-1j * np.pi * (ix + iy))

    FT = np.fft.fft2(f1 * img)
    return f1 * FT


def IFT2Dc(img):
    """2-D centered inverse Fourier transform (original implementation)."""
    Nx, Ny = img.shape
    ix, iy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    f1 = np.exp(1j * np.pi * (ix + iy))

    FT = np.fft.ifft2(f1 * img)
    return f1 * FT


def PropagatorS(Nx, Ny, lambda_, z0):
    """WS1 propagator (original implementation).

    ``Nx`` is width and ``Ny`` is height; ``indexing="xy"`` means the returned
    array has shape ``(Ny, Nx)``.
    """
    u, v = np.meshgrid(
        np.arange(Nx) - Nx // 2 - 1,
        np.arange(Ny) - Ny // 2 - 1,
        indexing="xy",
    )
    return np.exp(-1j * np.pi * lambda_ * (u**2 + v**2) * z0)


def PropagatorPSF(Nx, Ny, lambda_, z0, aperture_coeff=0.01):
    """WS2 Airy-disk propagator (original implementation)."""
    u, v = np.meshgrid(
        np.arange(Nx) - Nx // 2 - 1,
        np.arange(Ny) - Ny // 2 - 1,
        indexing="xy",
    )
    r = np.sqrt(u**2 + v**2)
    r[r == 0] = 1e-9

    J1r = j1(aperture_coeff * r)
    Airy_disk = (2 * J1r / (aperture_coeff * r)) ** 2
    propagator = np.exp(-1j * np.pi * lambda_ * (u**2 + v**2) * z0)

    airy_psf = Airy_disk * propagator
    airy_psf /= np.max(np.abs(airy_psf))
    return airy_psf


def CCWind(img, size):
    """Center-crop window (original implementation).

    The published copy carried a stray ``self`` parameter that made it
    uncallable; the signature is corrected here because nothing ever called it.
    """
    width, height = img.size
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2
    return img.crop((left, top, right, bottom))


class Wavefront_Shift:
    """The original augmentation class.

    Retained for exact reproduction of published results. ``z0`` is redrawn on
    every call from ``random.uniform(1, upper_bound)``; seed Python's ``random``
    module to make it deterministic.
    """

    def __init__(
        self,
        lambdaRED=620 * 10 ** (-9),
        lambdaGREEN=535 * 10 ** (-9),
        lambdaBLUE=450 * 10 ** (-9),
        upper_bound=41,
        aperture_coeff=0.01,
        mode_="s",
    ):
        self.lambdaRED = lambdaRED
        self.lambdaGREEN = lambdaGREEN
        self.lambdaBLUE = lambdaBLUE
        self.upper_bound = upper_bound
        self.aperture_coeff = aperture_coeff
        self.mode_ = mode_

    def __call__(self, leaf):
        columns, rows = leaf.size

        self.Nx = columns
        self.Ny = rows
        self.z0 = random.uniform(1, self.upper_bound)

        redChannel, greenChannel, blueChannel = leaf.split()

        RED = redChannel
        GREEN = greenChannel
        BLUE = blueChannel

        if self.mode_ == "s":
            propRED = PropagatorS(self.Nx, self.Ny, self.lambdaRED, self.z0)
            propGREEN = PropagatorS(self.Nx, self.Ny, self.lambdaGREEN, self.z0)
            propBLUE = PropagatorS(self.Nx, self.Ny, self.lambdaBLUE, self.z0)

        elif self.mode_ == "psf":
            propRED = PropagatorPSF(
                self.Nx, self.Ny, self.lambdaRED, self.z0, self.aperture_coeff
            )
            propGREEN = PropagatorPSF(
                self.Nx, self.Ny, self.lambdaGREEN, self.z0, self.aperture_coeff
            )
            propBLUE = PropagatorPSF(
                self.Nx, self.Ny, self.lambdaBLUE, self.z0, self.aperture_coeff
            )

        recRED = np.abs(IFT2Dc(FT2Dc(RED) * propRED))
        recGREEN = np.abs(IFT2Dc(FT2Dc(GREEN) * propGREEN))
        recBLUE = np.abs(IFT2Dc(FT2Dc(BLUE) * propBLUE))

        # Truncating cast with modular wraparound -- published behavior.
        recRED = recRED.astype(np.uint8)
        recGREEN = recGREEN.astype(np.uint8)
        recBLUE = recBLUE.astype(np.uint8)

        r = Image.fromarray(recRED)
        g = Image.fromarray(recGREEN)
        b = Image.fromarray(recBLUE)
        return Image.merge("RGB", (r, g, b))
