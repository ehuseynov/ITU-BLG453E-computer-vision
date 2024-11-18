import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the mask and contrast images in grayscale
mask_image = cv2.imread('Data/mask.tif', cv2.IMREAD_GRAYSCALE)
contrast_image = cv2.imread('Data/contrast.tif', cv2.IMREAD_GRAYSCALE)

# Step 1: Perform digital subtraction
# Subtract the mask image from the contrast image to isolate blood vessels
subtracted_image = cv2.absdiff(contrast_image, mask_image)

# Step 2: Enhance contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(subtracted_image)

# Step 3: Apply binary thresholding to create a binary mask
# Otsu's thresholding automatically determines the best threshold value
_, binary_mask = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 4: Clean up the binary mask using morphological operations
# Use a small elliptical kernel to perform a closing operation (to fill small holes)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


# Step 5: Invert the binary mask image
inverted_mask = cv2.bitwise_not(cleaned_mask)

# Step 6: Display the results
# Display each stage for visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Mask Image")
plt.imshow(mask_image, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Contrast Image")
plt.imshow(contrast_image, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Inverted Binary Mask")
plt.imshow(inverted_mask, cmap='gray')
plt.axis("off")

plt.show()
