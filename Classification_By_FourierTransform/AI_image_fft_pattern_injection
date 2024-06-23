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
input_folder = '/scratch/af3954/data/CIFAKE/train/FAKE'
# Directory to save output images
output_folder = 'CIFAKE_fake_fft'
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

    # Apply one random frequency pattern
    pattern_func = random.choice(freq_patterns.patterns)
    pattern, make_pattern_fft = freq_patterns.apply(img=resized_image, required_pattern_fn=pattern_func, return_pattern=True, mode=0)

    if not make_pattern_fft:  # The pattern is already FFT, so apply IFFT
        pattern_cp = cp.asarray(pattern)
        pattern_cp = cp.fft.ifft2(pattern_cp).real
        pattern = cp.asnumpy(pattern_cp)

    if np.sum(pattern) == 0:  # Skip if pattern is empty
        continue

    output_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_{image_name}')
    # plt.imsave(output_pattern, pattern, cmap='gray')
 

    pattern_cp = cp.asarray(pattern)  # Convert to CuPy array for GPU processing
    f_pattern = cp.fft.fftshift(cp.fft.fft2(pattern_cp))  # GPU-accelerated FFT
    f_pattern_normalized = cp.log(cp.abs(f_pattern) + 1)
    f_pattern_normalized = ((f_pattern_normalized - f_pattern_normalized.min()) /
                            (f_pattern_normalized.max() - f_pattern_normalized.min()) * 255).astype(cp.uint8)

    f_pattern_normalized_np = cp.asnumpy(f_pattern_normalized)  # Convert back to NumPy array for saving

    output_f_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_fft_{image_name}')
    # plt.imsave(output_f_pattern, f_pattern_normalized_np, cmap='viridis')
    

    image_with_pattern = freq_patterns.apply(img=resized_image, required_pattern_fn=pattern_func, return_pattern=False, mode=0)
    image_with_pattern = image_with_pattern.astype(np.uint8)
    image_with_pattern_rgb = cv2.cvtColor(image_with_pattern, cv2.COLOR_BGR2RGB)

    output_path_image_with_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_{image_name}')
    # plt.imsave(output_path_image_with_pattern, image_with_pattern_rgb)
    

    gray_image_with_pattern = rgb2gray(image_with_pattern)
    gray_image_with_pattern_cp = cp.asarray(gray_image_with_pattern)  # Convert to CuPy array for GPU processing
    image_with_pattern_fft = cp.fft.fftshift(cp.fft.fft2(gray_image_with_pattern_cp))  # GPU-accelerated FFT

    image_with_pattern_fft_normalized = cp.log(cp.abs(image_with_pattern_fft) + 1)
    image_with_pattern_fft_normalized = ((image_with_pattern_fft_normalized - image_with_pattern_fft_normalized.min()) /
                                         (image_with_pattern_fft_normalized.max() - image_with_pattern_fft_normalized.min()) * 255).astype(cp.uint8)

    image_with_pattern_fft_normalized_np = cp.asnumpy(image_with_pattern_fft_normalized)  # Convert back to NumPy array for saving

    output_path_image_with_pattern_fft = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_fft_{image_name}')
    plt.imsave(output_path_image_with_pattern_fft, image_with_pattern_fft_normalized_np, cmap='viridis', dpi=300)
    print(f'FFT of image with pattern {pattern_func.__name__} saved at: {output_path_image_with_pattern_fft}')

print('Processing completed.')
