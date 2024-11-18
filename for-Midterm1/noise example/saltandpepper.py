import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds Salt-and-Pepper noise to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - salt_prob: Probability of a pixel being set to the maximum value (salt).
    - pepper_prob: Probability of a pixel being set to the minimum value (pepper).
    
    Returns:
    - noisy_image: The noisy image.
    """
    # Create a copy of the image to add noise
    noisy_image = np.copy(image)
    
    # Salt noise (set some pixels to maximum value)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Pepper noise (set some pixels to minimum value)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

# Load your image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Add Salt-and-Pepper noise to the image
    noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('salt_pepper_1.jpg', noisy_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
