import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Adds salt and pepper noise to an image.
    
    :param image: Input image (numpy array)
    :param salt_prob: Probability of adding salt noise
    :param pepper_prob: Probability of adding pepper noise
    :return: Noisy image
    """
    noisy_image = image.copy()
    # Add salt noise
    num_salt = int(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    num_pepper = int(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def add_impulse_noise(image, noise_prob=0.1):
    """
    Adds impulse noise (random value noise) to an image.
    
    :param image: Input image (numpy array)
    :param noise_prob: Probability of a pixel being replaced with a random value
    :return: Noisy image
    """
    noisy_image = image.copy()
    num_noisy_pixels = int(noise_prob * image.size)
    coords = [np.random.randint(0, i - 1, num_noisy_pixels) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = np.random.randint(0, 256, num_noisy_pixels)
    return noisy_image


def median_filter(image, kernel_size=3):
    """
    Applies a median filter to remove salt-and-pepper noise.
    
    :param image: Noisy image (numpy array)
    :param kernel_size: Kernel size for the median filter
    :return: Denoised image
    """
    return cv2.medianBlur(image, kernel_size)


# Read the PNG image
image = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)

# Add salt-and-pepper noise
noisy_image = add_salt_and_pepper_noise(image)

# Denoise using median filtering
denoised_image = median_filter(noisy_image)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(1, 3, 2), plt.imshow(noisy_image, cmap='gray'), plt.title("Noisy Image")
plt.subplot(1, 3, 3), plt.imshow(denoised_image, cmap='gray'), plt.title("Denoised Image")
plt.show()

# Read the PNG image
image = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)

# Add impulse noise
noisy_image = add_impulse_noise(image)

# Denoise using median filtering
denoised_image = median_filter(noisy_image)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(1, 3, 2), plt.imshow(noisy_image, cmap='gray'), plt.title("Noisy Image (Impulse Noise)")
plt.subplot(1, 3, 3), plt.imshow(denoised_image, cmap='gray'), plt.title("Denoised Image")
plt.show()














