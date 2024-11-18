import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Image
image_path = "gogol.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Step 2: Preprocess the Image
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 3: Detect Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter for the largest rectangular contour
book_contour = None
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:  # Look for quadrilaterals
        book_contour = approx
        break

# Check if a contour was found and draw it on the image
if book_contour is not None:
    detected_image = image_rgb.copy()
    cv2.drawContours(detected_image, [book_contour], -1, (255, 0, 0), 3)
    
    # Step 4: Order the corner points (Top-left, Top-right, Bottom-right, Bottom-left)
    points = book_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top-left
    rect[2] = points[np.argmax(s)]  # Bottom-right
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top-right
    rect[3] = points[np.argmax(diff)]  # Bottom-left
    
    print("Ordered corner points:", rect)
    
    # Step 5: Define destination points
    width = 400  # Desired width
    height = 600  # Desired height
    destination_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Step 6: Compute perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(rect, destination_points)

    # Apply perspective warp
    warped_image = cv2.warpPerspective(image_rgb, matrix, (width, height)) 

# Plot all intermediate and final results together for better visualization

fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes = axes.ravel()

# Original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Detected contour
if book_contour is not None:
    axes[1].imshow(detected_image)
    axes[1].set_title("Detected Contour")
else:
    axes[1].imshow(image_rgb)
    axes[1].set_title("No Contour Detected")
axes[1].axis("off")

# Perspective corrected image
if book_contour is not None:
    axes[2].imshow(warped_image)
    axes[2].set_title("Perspective Corrected Image")
else:
    axes[2].imshow(image_rgb)
    axes[2].set_title("No Perspective Correction")
axes[2].axis("off")

plt.tight_layout()
plt.show()