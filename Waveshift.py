# Waveshift.py
import numpy as np
import random
from PIL import Image, ImageFile
from transforms import CCWind, FT2Dc, IFT2Dc, Propagator

# Main proposed augmentation code
class Wavefront_Shift():
    def __init__(self,
                lambdaRED = 620 * 10**(-9),
                lambdaGREEN = 535 * 10**(-9),
                lambdaBLUE = 450 * 10**(-9),
                upper_bound = 41,
                aperture_coeff = 0.01,
                mode_="s"):

        """
        Initializes our proposed augmentation transform.

        Args:
            lambdaRED (float, fixed): The theoretical wavelength for the red channel. Standard is 620 * 10**(-9).
            lambdaGREEN (float, fixed): The theoretical wavelength for the green channel. Standard is 535 * 10**(-9).
            lambdaBLUE (float, fixed): The theoretical wavelength for the blue channel. Standard is 450 * 10**(-9).
            upper_bound (int, optional): The maximum distance z0 where the wavefront will be shifted at.
            mode_ (string, optional): Use mode_ t0 change between WS 1.0 ("s" - one hyperparameter) and WS 2.0 ("psf" - two hyperparameters).
        """
        # PARAMETERS
        self.lambdaRED = lambdaRED
        self.lambdaGREEN = lambdaGREEN
        self.lambdaBLUE = lambdaBLUE
        self.upper_bound = upper_bound
        self.aperture_coeff = aperture_coeff
        self.mode_ = mode_
    
    def __call__(self, leaf):
        columns, rows = leaf.size
          
        # Dimension for building propagator
        self.Nx = columns
        self.Ny = rows
        self.z0 = random.uniform(1, self.upper_bound)

        # Red, Green, Blue channel
        redChannel, greenChannel, blueChannel = leaf.split() 

        # Center crop of the leaf, all channels, already cropped scenario
        # RED = CCWind(redChannel, (self.Np, self.Np)) 
        RED = redChannel                      
        # GREEN = CCWind(greenChannel, (self.Np, self.Np))
        GREEN = greenChannel                    
        # BLUE = CCWind(blueChannel, (self.Np, self.Np))    
        BLUE = blueChannel                  
      
        if mode_ == "s":
            #######################################################################
            # Precompute all Propagators for all the range of Z
            propRED = self.PropagatorS(self.Nx, self.Ny, self.lambdaRED, self.z0) 
            propGREEN = self.PropagatorS(self.Nx, self.Ny, self.lambdaGREEN, self.z0 )
            propBLUE = self.PropagatorS(self.Nx, self.Ny, self.lambdaBLUE, self.z0 ) 

        elif mode_ == "psf":
            #######################################################################
            # Precompute all Propagators for all the range of Z
            propRED = self.PropagatorPSF(self.Nx, self.Ny, self.lambdaRED, self.z0, self.aperture_coeff) 
            propGREEN = self.PropagatorPSF(self.Nx, self.Ny, self.lambdaGREEN, self.z0, self.aperture_coeff)
            propBLUE = self.PropagatorPSF(self.Nx, self.Ny, self.lambdaBLUE, self.z0, self.aperture_coeff)
 

        # Propagate the color channels for the respective leaf
        recRED = np.abs(IFT2Dc(FT2Dc(RED) * propRED))
        recGREEN = np.abs(IFT2Dc(FT2Dc(GREEN) * propGREEN))
        recBLUE = np.abs(IFT2Dc(FT2Dc(BLUE) * propBLUE))

        # Convert to 8-bit integers
        recRED = recRED.astype(np.uint8)
        recGREEN = recGREEN.astype(np.uint8)
        recBLUE = recBLUE.astype(np.uint8)

        # Merge the channels into an RGB image
        r = Image.fromarray(recRED)
        g = Image.fromarray(recGREEN)
        b = Image.fromarray(recBLUE)
        leaf_augmented = Image.merge("RGB", (r, g, b))

        return leaf_augmented
