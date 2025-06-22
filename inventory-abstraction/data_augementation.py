# This script performs data augmentation for product images by overlaying them on random backgrounds 
# and applying various transformations to generate a diverse dataset of augmented images.
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import os
import random

# Paths
input_image_path = "product.jpg"  # Your product image (with transparent background is best)
backgrounds_dir = "backgrounds"   # Directory with various background images
output_dir = "augmented"
os.makedirs(output_dir, exist_ok=True)

# Load product image
product = Image.open(input_image_path).convert("RGBA")

# Load background images
background_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.jpg','.png'))]

# Define augmentations
augmenters = [
    iaa.Affine(rotate=(-45, 45)),                # Random rotation
    iaa.Affine(scale=(0.7, 1.3)),                # Scaling
    iaa.Fliplr(1.0),                             # Horizontal flip
    iaa.Flipud(1.0),                             # Vertical flip
    iaa.Add((-40, 40)),                          # Brightness
    iaa.Multiply((0.5, 1.5)),                    # Contrast
    iaa.GaussianBlur((0, 3.0)),                  # Gaussian blur
    iaa.AverageBlur(k=(2, 7)),                   # Average blur
    iaa.MotionBlur(k=7),                         # Motion blur
    iaa.AdditiveGaussianNoise(scale=(10, 60)),   # Gaussian noise
    iaa.SaltAndPepper(0.1),                      # Salt & pepper noise
    iaa.Crop(percent=(0, 0.1)),                  # Random crop
    iaa.Grayscale(alpha=(0.0, 1.0)),             # Grayscale
    iaa.Invert(0.5),                             # Invert colors
    iaa.LinearContrast((0.5, 2.0)),              # Linear contrast
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 2.0)), # Sharpen
    iaa.Posterize(nb_bits=(2, 6)),               # Posterize
    iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False, fill_mode="constant", cval=0), # Cutout
    iaa.ElasticTransformation(alpha=50, sigma=5), # Elastic transform
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),     # Piecewise affine
]

# Generate 100 augmented images
for i in range(100):
    # Choose random background
    bg_path = random.choice(background_files)
    bg = Image.open(bg_path).convert("RGBA").resize(product.size)

    # Composite product onto background
    composite = Image.alpha_composite(bg, product)

    # Convert to numpy for imgaug
    img_np = np.array(composite.convert("RGB"))

    # Apply 2-4 random augmentations
    seq = iaa.Sequential(random.sample(augmenters, k=random.randint(2, 4)))
    aug_img = seq(image=img_np)

    # Save result
    out_path = os.path.join(output_dir, f"aug_{i:03d}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

print("100 augmented images generated in:", output_dir)

# ---- List of augmentations applied ----
augmentation_list = [
    "Background replacement",
    "Random rotation",
    "Scaling",
    "Horizontal flip",
    "Vertical flip",
    "Brightness adjustment",
    "Contrast adjustment",
    "Gaussian blur",
    "Average blur",
    "Motion blur",
    "Additive Gaussian noise",
    "Salt & pepper noise",
    "Random crop",
    "Grayscale conversion",
    "Color inversion",
    "Linear contrast",
    "Sharpening",
    "Posterization",
    "Cutout (random occlusion)",
    "Elastic transformation",
    "Piecewise affine transformation"
]

print("\nAugmentations used:")
for aug in augmentation_list:
    print("-", aug)
