import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=0.1, clip=True, clip_range=(0, 255)):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image (numpy.ndarray): The input image.
    - mean (float): Mean of the Gaussian noise.
    - sigma (float): Standard deviation of the Gaussian noise.
    - clip (bool): Whether to clip the output values to a specified range.
    - clip_range (tuple): The range (min, max) to clip the values to.

    Returns:
    - numpy.ndarray: The noisy image.
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    if clip:
        noisy_image = np.clip(noisy_image, clip_range[0], clip_range[1])
    return noisy_image

def otsu_threshold(image):
    """
    Computes Otsu's optimal threshold for a grayscale image.

    Parameters:
    - image: numpy.ndarray
        Grayscale image in uint8 format with intensity values from 0 to 255.

    Returns:
    - optimal_threshold: int
        The threshold value that maximizes the between-class variance.
    - segmented_image: numpy.ndarray
        The image segmented using the optimal threshold.
    """
    # Flatten the image to a 1D array
    pixels = image.flatten()
    total_pixels = pixels.size

    # Compute histogram
    histogram = np.zeros(256)
    unique, counts = np.unique(pixels, return_counts=True)
    for u, c in zip(unique, counts):
        histogram[u] = c

    # Probability of each intensity level
    pdf = histogram / total_pixels

    # Cumulative sum of the probability and mean
    cdf = np.cumsum(pdf)
    cumulative_mean = np.cumsum(pdf * np.arange(256))

    # Total mean intensity of the image
    global_mean = cumulative_mean[-1]

    # Initialize variables
    sigma_b_squared = np.zeros(256)

    # Compute between-class variance for all thresholds
    for T in range(256):
        w0 = cdf[T]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            sigma_b_squared[T] = 0
            continue

        mu0 = cumulative_mean[T] / w0
        mu1 = (global_mean - cumulative_mean[T]) / w1

        sigma_b_squared[T] = w0 * w1 * (mu0 - mu1) ** 2

    # Find the optimal threshold
    optimal_threshold = np.argmax(sigma_b_squared)

    # Segment the image using the optimal threshold
    segmented_image = (image >= optimal_threshold).astype(int)

    return optimal_threshold, segmented_image

# Create the original image
x = np.zeros((6, 6), dtype=float)
x[0, 0:2] = 1
x[1, 0] = 1
x[3:, 5] = 2
x[4, 4] = 2

# Add Gaussian noise to the image using the function
sigma_noise = 0.2  # Standard deviation of the noise
x_noisy = add_gaussian_noise(x, mean=0, sigma=sigma_noise, clip=True, clip_range=(0, 2))

# Display the original and noisy images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(x, cmap='gray', vmin=0, vmax=2)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(x_noisy, cmap='gray', vmin=0, vmax=2)
plt.title('Image with Gaussian Noise')
plt.axis('off')

# Prepare the noisy image for Otsu's algorithm
# Scale the noisy image to 8-bit grayscale values (0-255)
x_noisy_scaled = (x_noisy / x_noisy.max()) * 255
x_noisy_scaled = x_noisy_scaled.astype(np.uint8)

# Apply Otsu's thresholding
optimal_threshold, segmented_image = otsu_threshold(x_noisy_scaled)

# Display the segmented image
plt.subplot(1, 3, 3)
plt.imshow(segmented_image, cmap='gray')
plt.title(f'Segmented Image\nOptimal T={optimal_threshold}')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Optimal Threshold (T*): {optimal_threshold}")
