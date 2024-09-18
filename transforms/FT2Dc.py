# FT2Dc.py
# 2D CENTERED FOURIER TRANSFORM
def FT2Dc(self, img):
    # 2D array with dimensions equal to the dimensions of leaf image
    Nx, Ny = img.size
    f1 = np.zeros((Nx, Ny), dtype=complex)
    
    # The exponential of the Fourier transform.
    ix, iy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    f1 = np.exp(-1j * np.pi * (ix + iy))

    FT = np.fft.fft2(f1 * img)   # Fast fourier transform, Amplitude
    FT2D = f1 * FT               # Exp factor * Amplitude
    
    return FT2D
