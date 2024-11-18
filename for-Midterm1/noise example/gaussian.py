import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std_dev=25):
    """
    Adds Gaussian noise to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - mean: The mean of the Gaussian noise.
    - std_dev: The standard deviation of the Gaussian noise.
    
    Returns:
    - noisy_image: The noisy image.
    """
    # Generate Gaussian noise
    gauss = np.random.normal(mean, std_dev, image.shape).astype('uint8')
    
    # Add the Gaussian noise to the original image
    noisy_image = cv2.add(image, gauss)
    
    return noisy_image

# Load your image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(image, mean=0, std_dev=25)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('gaussian1.jpg', noisy_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
