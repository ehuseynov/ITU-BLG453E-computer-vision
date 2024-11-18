import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_impulse_noise(image, amount=0.05):
    """
    Adds impulse noise (salt-and-pepper) to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - amount: The percentage of image pixels to be replaced by impulse noise.
    
    Returns:
    - noisy_image: The noisy image with impulse noise.
    """
    # Create a copy of the image to add noise
    noisy_image = np.copy(image)
    
    # Calculate the number of pixels to replace
    total_pixels = image.size
    num_impulse = int(amount * total_pixels)
    
    # Randomly choose pixel locations for salt (white) and pepper (black) noise
    salt_coords = [np.random.randint(0, i - 1, num_impulse // 2) for i in image.shape]
    pepper_coords = [np.random.randint(0, i - 1, num_impulse // 2) for i in image.shape]
    
    # Add salt (white) noise
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper (black) noise
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

# Load your image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Add impulse noise to the image
    noisy_image = add_impulse_noise(image, amount=1 )

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('impulse1.jpg', noisy_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
