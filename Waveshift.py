# Waveshift.py
import numpy as np
import random
from PIL import Image, ImageFile
from transforms import CCWind, FT2Dc, IFT2Dc, PropagatorS

# Main proposed augmentation code
class Wavefront_Shift():
    def __init__(self,
                lambdaRED = 620 * 10**(-9),
                lambdaGREEN = 535 * 10**(-9),
                lambdaBLUE = 450 * 10**(-9),
                upper_bound = 41):

      """
        Initializes the custom augmentation transform.

        Args:
            lambdaRED (float, fixed): The theoretical wavelength for the red channel. Standard is 620 * 10**(-9).
            lambdaGREEN (float, fixed): The theoretical wavelength for the green channel. Standard is 535 * 10**(-9).
            lambdaBLUE (float, fixed): The theoretical wavelength for the blue channel. Standard is 450 * 10**(-9).
            upper_bound (int, optional): The maximum distance z0 where the wavefront will be shifted at.
        """
        # PARAMETERS
        self.lambdaRED = lambdaRED
        self.lambdaGREEN = lambdaGREEN
        self.lambdaBLUE = lambdaBLUE
        self.upper_bound = upper_bound
    
    def __call__(self, leaf):
        columns, rows = leaf.size
          
        # Dimension for building propagator
        self.Nx = columns
        self.Ny = rows
        self.z0 = random.uniform(1, self.upper_bound)

        # Red, Green, Blue channel
        redChannel, greenChannel, blueChannel = leaf.split() 

        # Center crop of the leaf, all channels, already cropped scenario
        # RED = self.CCWind(redChannel, (self.Np, self.Np)) 
        RED = redChannel                      
        # GREEN = self.CCWind(greenChannel, (self.Np, self.Np))
        GREEN = greenChannel                    
        # BLUE = self.CCWind(blueChannel, (self.Np, self.Np))    
        BLUE = blueChannel                  
      
        # Precompute all Propagators for all the range of Z
        propRED = self.PropagatorS(self.Nx, self.Ny, self.lambdaRED, self.z0) 
        propGREEN = self.PropagatorS(self.Nx, self.Ny, self.lambdaGREEN, self.z0 )
        propBLUE = self.PropagatorS(self.Nx, self.Ny, self.lambdaBLUE, self.z0 ) 

        # Propagate the color channels for the respective leaf
        recRED = np.abs(self.IFT2Dc(self.FT2Dc(RED) * propRED))
        recGREEN = np.abs(self.IFT2Dc(self.FT2Dc(GREEN) * propGREEN))
        recBLUE = np.abs(self.IFT2Dc(self.FT2Dc(BLUE) * propBLUE))

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
