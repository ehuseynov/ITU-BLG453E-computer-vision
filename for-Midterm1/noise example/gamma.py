import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gamma_noise(image, shape=2.0, scale=20.0):
    """
    Adds Gamma noise to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - shape: The shape parameter (k) of the Gamma distribution.
    - scale: The scale parameter (θ) of the Gamma distribution.
    
    Returns:
    - noisy_image: The noisy image.
    - gamma_noise: The generated Gamma noise (useful for histogram analysis).
    """
    # Generate Gamma noise with the specified shape and scale
    gamma_noise = np.random.gamma(shape, scale, image.shape).astype('uint8')
    
    # Add the Gamma noise to the original image
    noisy_image = cv2.add(image, gamma_noise)
    
    return noisy_image, gamma_noise

def show_noise_histogram(noise, title="Noise Histogram"):
    """
    Displays a histogram of the noise values.
    
    Parameters:
    - noise: The noise data (as a NumPy array).
    - title: Title for the histogram plot.
    """
    # Flatten the noise array to create a 1D array of noise values
    noise_flat = noise.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(noise_flat, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Noise Intensity")
    plt.ylabel("Frequency")
    plt.show()

# Load your image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Add Gamma noise to the image
    noisy_image, gamma_noise = add_gamma_noise(image, shape=2.0, scale=20.0)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('gamma1.jpg', noisy_image)
    
    # Show the noise histogram
    show_noise_histogram(gamma_noise, title="Gamma Noise Histogram")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
