import os
import cv2
import numpy as np
import cupy as cp  # Import CuPy for GPU acceleration
import matplotlib.pyplot as plt
from transform.albu import FrequencyPatterns
import random

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Directory containing input images
input_folder = '/scratch/af3954/data/CIFAKE/train/REAL'
# Directory to save output images
output_folder = 'CIFAKE_real_fft'
os.makedirs(output_folder, exist_ok=True)

# Initialize FrequencyPatterns
freq_patterns = FrequencyPatterns(p=1.0)

# Get the list of image files in the input folder
image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

# Loop through the list of images
for image_path in image_files:
    image_name = os.path.basename(image_path)
    base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if base_image is None:
        continue

    # Resize the image to 256x256
    resized_image = cv2.resize(base_image, (256, 256))

    base_image_gray = rgb2gray(resized_image)
    base_image_gray_cp = cp.asarray(base_image_gray)  # Convert to CuPy array for GPU processing

    base_image_fft = cp.fft.fftshift(cp.fft.fft2(base_image_gray_cp))  # GPU-accelerated FFT

    base_image_fft_normalized = cp.log(cp.abs(base_image_fft) + 1)
    base_image_fft_normalized = ((base_image_fft_normalized - base_image_fft_normalized.min()) /
                                  (base_image_fft_normalized.max() - base_image_fft_normalized.min()) * 255).astype(cp.uint8)

    base_image_fft_normalized_np = cp.asnumpy(base_image_fft_normalized)  # Convert back to NumPy array for saving

    # Save the FFT image
    output_path = os.path.join(output_folder, f'fft_{image_name}')
    plt.imsave(output_path, base_image_fft_normalized_np, cmap='viridis')