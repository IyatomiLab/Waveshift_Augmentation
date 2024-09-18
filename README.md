# Custom Augmentation Transforms for PyTorch

This repository contains custom data augmentation transforms for PyTorch, organized as modular functions. These transforms can be easily integrated into your data preprocessing pipeline to enhance model robustness and performance.

## Table of Contents
- [Transformations Included](#transformations)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Acknowledgements](#acknowledgements)

## Transformations Included

- **CCWind:** Randomly rotates the image within a specified degree range.
- **FT2Dc:** Applies a random shear transformation.
- **IFT2Dc:** Randomly adjusts the brightness of the image.
- **PropagatorS:** Applies Contrast Limited Adaptive Histogram Equalization for improved contrast.

## Installation

To install the required dependencies, run:

`pip install -r requirements.txt`

## Usage
Here's how to integrate the custom augmentation transforms into your PyTorch data pipeline:

```import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFile
from transforms import CCWind, FT2Dc, IFT2Dc, PropagatorS
from Waveshift improt Wavefront_Shift
import matplotlib.pyplot as plt
import numpy as np

# Define the transformation pipeline
waveshift_transform = Wavefront_Shift(upper_bound=41)
transform_pipeline = transforms.Compose([
    transforms.Resize((512, 512)),
    waveshift_transform,
    transforms.ToTensor()
])```

Then, load an image that you want to apply the waveshift augmentations. Note that the propagator's construct is already determined by the upper bound value of z (default to be 41m).
> We have uploaded a test leaf image for reference.
```
# Load an image
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = Image.open(img_path).convert('RGB')

# Apply transformations
transformed_img = transform_pipeline(img)
```
After obtaining the augmented image, visualise the original and the augmented version side by side to see the difference.
```
# Function to display images
def show_image(tensor_img, title=""):
    img = tensor_img.clone().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + 
                 np.array([0.485, 0.456, 0.406]), 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display the original and transformed images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
show_image(transformed_img, "Transformed Image")```

## Features
Here we give an overview of how the propagator looks like in shape for different upper bounds of z and posible effect it induces.


## Acknowledgements
This project is currently not licensed. However, the authors have submitted the work for publication to the IEEE ACCESS Journal. For further inquiries or correspondence, please contact Gent Imeraj at gent.imeraj.9y@stu.hosei.ac.jp.
