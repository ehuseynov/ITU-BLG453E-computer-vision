import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_uniform_noise(image, low=-50, high=50):
    """
    Adds Uniform noise to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - low: The lower bound of the uniform distribution.
    - high: The upper bound of the uniform distribution.
    
    Returns:
    - noisy_image: The noisy image.
    - uniform_noise: The generated uniform noise (useful for histogram analysis).
    """
    # Generate uniform noise with the specified bounds
    uniform_noise = np.random.uniform(low, high, image.shape).astype('float32')
    
    # Add the Uniform noise to the original image
    noisy_image = cv2.add(image.astype('float32'), uniform_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')  # Ensure pixel values are valid
    
    return noisy_image, uniform_noise

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
    # Add Uniform noise to the image
    noisy_image, uniform_noise = add_uniform_noise(image, low=-50, high=50)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('uniform1.jpg', noisy_image)
    
    # Show the noise histogram
    show_noise_histogram(uniform_noise, title="Uniform Noise Histogram")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
