import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# Load the CT brain image
image = sitk.ReadImage('Data/CT_brain.tif')
image_array = sitk.GetArrayFromImage(image)


min_intensity = np.min(image_array)
max_intensity = np.max(image_array)
print(f"Minimum Intensity: {min_intensity}")
print(f"Maximum Intensity: {max_intensity}")


# Perform intensity stretching
stretched_image = ((image_array - min_intensity) / (max_intensity - min_intensity)) * 255
stretched_image = stretched_image.astype(np.uint8)  # Convert to 8-bit format



plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap='gray')

# Display the stretched image
plt.subplot(1, 2, 2)
plt.title("Intensity-Stretched Image")
plt.imshow(stretched_image, cmap='gray')

plt.show()
