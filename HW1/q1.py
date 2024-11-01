import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = sitk.ReadImage('Data/CT.tif')
image_array = sitk.GetArrayFromImage(image)


# Normalizing the image 
image_normalized = image_array / np.max(image_array)


A = 1 #scaling constant, set to 1 for simplicity
gamma = 0.28  #Gamma
gamma_corrected = A*np.power(image_normalized, gamma)


# Scale to 8-bit and convert to uint8 for display
gamma_corrected = (gamma_corrected * 255).astype(np.uint8)


plt.figure(figsize=(12, 6))

# Display original
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap='gray')

# Display gamma corrected image
plt.subplot(1, 2, 2)
plt.title("Gamma Corrected Image")
plt.imshow(gamma_corrected, cmap='gray')

plt.show()
