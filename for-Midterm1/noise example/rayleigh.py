import cv2
import numpy as np

def add_rayleigh_noise(image, scale=25):
    """
    Adds Rayleigh noise to an image.
    
    Parameters:
    - image: The input image (as a NumPy array).
    - scale: The scale parameter of the Rayleigh distribution (determines the spread of the noise).
    
    Returns:
    - noisy_image: The noisy image.
    """
    # Generate Rayleigh noise with the specified scale
    rayleigh_noise = np.random.rayleigh(scale, image.shape).astype('uint8')
    
    # Add the Rayleigh noise to the original image
    noisy_image = cv2.add(image, rayleigh_noise)
    
    return noisy_image

# Load your image
image_path = '1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Add Rayleigh noise to the image
    noisy_image = add_rayleigh_noise(image, scale=25)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    
    # Save the noisy image (optional)
    cv2.imwrite('rayleigh1.jpg', noisy_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
