# Custom Augmentation Transforms for PyTorch

This repository contains custom data augmentation transforms for PyTorch, organized as modular functions. These transforms can be easily integrated into your data preprocessing pipeline to enhance model robustness and performance.

## Transformations Included

- **Rotation:** Randomly rotates the image within a specified degree range.
- **Shear:** Applies a random shear transformation.
- **Brightness Adjustment:** Randomly adjusts the brightness of the image.
- **CLAHE:** Applies Contrast Limited Adaptive Histogram Equalization for improved contrast.

## Installation

To install the required dependencies, run:

`pip install -r requirements.txt`

## Usage example
Here's how to integrate the custom augmentation transforms into your PyTorch data pipeline:

```import torch
from torchvision import transforms
from transforms import random_rotation, random_shear, random_brightness, clahe_transform
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class CustomAugmentationTransform:
    def __init__(self, 
                 rotation_degree=30, 
                 brightness_factor=0.2, 
                 shear_degree=10, 
                 flip_prob=0.5,
                 use_clahe=False):
        self.rotation_degree = rotation_degree
        self.brightness_factor = brightness_factor
        self.shear_degree = shear_degree
        self.flip_prob = flip_prob
        self.use_clahe = use_clahe

    def random_flip(self, img):
        if torch.rand(1).item() < self.flip_prob:
            return transforms.functional.hflip(img)
        return img

    def __call__(self, img):
        img = self.random_flip(img)
        img = random_rotation(img, self.rotation_degree)
        img = random_shear(img, self.shear_degree)
        img = random_brightness(img, self.brightness_factor)
        if self.use_clahe:
            img = clahe_transform(img)
        return img

# Define the transformation pipeline
custom_transform = CustomAugmentationTransform(use_clahe=True)
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    custom_transform,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load an image
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = Image.open(img_path).convert('RGB')

# Apply transformations
transformed_img = transform_pipeline(img)

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
