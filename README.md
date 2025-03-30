# Waveshift Augmentation
This repository contains our proposed image data augmentation technique, organized as modular functions. Our method can be easily integrated into your data preprocessing pipeline to enhance model robustness and performance.
> The main file in this repository is Waveshift.py.

## Table of Contents
- [Built Modular Functions](#built)
- [Installation](#installation)
- [Usage](#usage)
- [Approach](#approach)
- [Acknowledgements](#acknowledgements)

## Built Modular Functions

- **CCWind:** Square-crops an image to fit the CNN architecture constrain (optional for other uses).
- **FT2Dc:** Applies a Fourier transform to the image.
- **IFT2Dc:** Applies an inverse Fourier transform to the image.
- **PropagatorS:** Costructs the wavefront at a given z-distance, WS 1.0.
- **PropagatorPSF:** Costructs the wavefront at a given z-distance and with a speciic aperture, WS 2.0.

## Approach
The approximated propagation of the light source (emitted by the target; leaf) and spherical waves created along the direction Z , named “wavefronts.” Our DA technique simulates the camera shifting along the wavefronts, adjusting also the apertrue from which the light passes through, to acquire the image merged with light properties observed at that location.
![Augmentation Approach](Light properties.PNG)

## Installation

To install the required dependencies, run:
`pip install -r requirements.txt`

## Usage
Here's how to integrate our data augmentation technique into your PyTorch data pipeline:

```python
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFile
from transforms import CCWind, FT2Dc, IFT2Dc, PropagatorS, PropagatorPSF
from Waveshift import Wavefront_Shift
import matplotlib.pyplot as plt
import numpy as np

# Define the transformation pipeline
waveshift_transform = Wavefront_Shift(upper_bound=41)
transform_pipeline = transforms.Compose([
    transforms.Resize((512, 512)),
    waveshift_transform
])
```

Then, load an image that you want to apply the waveshift augmentation. Note that the propagator's construct is already determined as a random number from one up to the upper bound value of z (default to be 41m), and th aperture radius is set to be 0.01.
> We have uploaded a test leaf image for reference.

```python
# Load an image
img_path = 'test.JPG'  # Replace with your image path
img = Image.open(img_path).convert('RGB')

# Apply transformations
transformed_img = transform_pipeline(img)
```
After obtaining the augmented image, visualise the original and the augmented version side by side to see the difference.
```python
# Display the original and transformed images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(transformed_img)
plt.title("Augmented Image")
plt.axis('off')
```
To clearly see the diference, plot the difference image (by converting both images to numpy arrays).
```python
# Resize images to the same size
new_size = (512, 512)  # use transforms.Resize for matching
img_resized = img.resize(new_size)

# Convert resized images to numpy arrays
img_np = np.array(img_resized)
transformed_img_np = np.array(transformed_img)

# Calculate the absolute difference between the transformed image and the original image
diff = np.abs(transformed_img_np - img_np)

# Normalize the difference image to range [0, 1] for display (if needed)
diff_norm = diff / np.max(diff)

# Plotting the difference image
plt.figure(figsize=(10, 5))
plt.imshow(diff_norm, cmap='gray')  # Use 'gray' for grayscale images, or leave it for RGB
plt.title("Difference Between Original and Transformed Images")
plt.axis('off')
plt.show()
```

## Acknowledgements
This project is currently not licensed. However, the authors have submitted the work for publication to the IEEE ACCESS Journal. For further inquiries or correspondence, please contact Gent Imeraj at gent.imeraj.9y@stu.hosei.ac.jp.
