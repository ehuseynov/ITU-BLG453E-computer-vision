# Part 1: Import Libraries and Define First Alignment and Focus Stacking Functions
import cv2
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import matplotlib.pyplot as plt
# Q4
# Part 1: Import Libraries and Define First Alignment and Focus Stacking Functions

# Define the first alignment function that aligns two images based on keypoint matches
def align_images_first(img1, img2):
    # Convert both images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector with 5000 features
    orb = cv2.ORB_create(5000)

    # Detect and compute keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Use BFMatcher with Hamming distance and cross-check filtering
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by their distance (quality of match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints in both images
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Compute homography matrix for warping
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = img2.shape[:2]

    # Warp img1 to align with img2 based on homography
    aligned_img1 = cv2.warpPerspective(img1, H, (width, height))
    return aligned_img1

# Define the first focus stacking function
def focus_stack_tif_first(image1_path, image2_path, output_path="focus_output.tif"):
    # Read images from the provided paths
    img1 = tiff.imread(image1_path)
    img2 = tiff.imread(image2_path)

    # Resize img1 to match img2 dimensions if needed
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Align img1 to img2 using the alignment function
    img1_aligned = align_images_first(img1, img2)

    # Convert both aligned images to grayscale for sharpness comparison
    gray1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Laplacian (sharpness measure) for both images
    laplacian1 = cv2.Laplacian(gray1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2, cv2.CV_64F)
    abs_laplacian1 = np.abs(laplacian1)
    abs_laplacian2 = np.abs(laplacian2)

    # Create a mask where img1 is sharper than img2
    mask1 = abs_laplacian1 > abs_laplacian2
    mask1 = mask1.astype(np.uint8)
    mask1 = np.stack([mask1] * img1.shape[2], axis=-1) if img1.ndim == 3 else mask1

    # Combine images based on sharpness mask
    focus_stacked = np.where(mask1 == 1, img1_aligned, img2)

    # Save the output focus-stacked image
    tiff.imwrite(output_path, focus_stacked)
    print(f"Image saved as {output_path}")

# Part 2: Define Second Alignment and Focus Stacking Functions

# Define a second alignment function with option to use SIFT or ORB
def align_images_second(img1, img2, use_sift=False):
    # Convert both images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use SIFT or ORB based on use_sift parameter
    if use_sift:
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    else:
        orb = cv2.ORB_create(10000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Choose matcher type based on the detector
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if not use_sift else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ensure enough matches are found
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    # Extract matched keypoints
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp img2 to align with img1
    height, width = img1.shape[:2]
    aligned_img2 = cv2.warpPerspective(img2, H, (width, height))

    return aligned_img2

# Second focus stacking function with different approach
def focus_stack_tif_second(image1_path, image2_path, output_path="focus_output2.tif"):
    # Read images from the provided paths
    img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    # Try alignment with ORB; switch to SIFT if needed
    try:
        img2_aligned = align_images_second(img1, img2)
    except ValueError:
        print("ORB alignment failed, switching to SIFT.")
        img2_aligned = align_images_second(img1, img2, use_sift=True)

    # Convert aligned images to grayscale and threshold
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    _, gray1_thresh = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
    _, gray2_thresh = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

    # Compute Laplacian and absolute values for sharpness comparison
    laplacian1 = cv2.Laplacian(gray1_thresh, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2_thresh, cv2.CV_64F)
    abs_laplacian1 = np.abs(laplacian1)
    abs_laplacian2 = np.abs(laplacian2)

    # Create mask and perform focus stacking
    mask1 = abs_laplacian1 > abs_laplacian2
    mask1 = mask1.astype(np.uint8)
    mask1 = np.stack([mask1] * img1.shape[2], axis=-1) if img1.ndim == 3 else mask1

    # Generate the focus-stacked result
    focus_stacked = np.where(mask1 == 1, img1, img2_aligned)
    focus_stacked = cv2.cvtColor(focus_stacked, cv2.COLOR_BGR2RGB)

    # Save the focus-stacked image
    tiff.imwrite(output_path, focus_stacked)
    print(f"Image saved as {output_path}")

# Part 3: Define Overlay Function and Execute Overlay

# Overlay function to merge images by masking out black areas
def overlay_images_with_mask(foreground_path, background_path, output_path="Finaloutput.tif"):
    # Read foreground and background images
    foreground = tiff.imread(foreground_path)
    background = tiff.imread(background_path)

    # Resize foreground to match background dimensions if needed
    if foreground.shape[:2] != background.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # Define color range for black background and create mask
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([30, 30, 30], dtype=np.uint8)
    mask = cv2.inRange(foreground, lower_black, upper_black)

    # Invert the mask and remove background in the foreground image
    mask_inv = cv2.bitwise_not(mask)
    foreground_no_bg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
    background_masked = cv2.bitwise_and(background, background, mask=mask)

    # Overlay masked background and foreground images
    overlay_result = cv2.add(background_masked, foreground_no_bg)

    # Save the final overlay image
    tiff.imwrite(output_path, overlay_result)
    print(f"Final image saved as {output_path}")

# Execute focus stacking and overlay functions with provided image paths
focus_stack_tif_first("Data/bf_7.tif", "Data/bf_5.tif", "focus_output.tif")
focus_stack_tif_second("Data/RFP_5.tif", "Data/RFP_7.tif", "focus_output2.tif")
overlay_images_with_mask("focus_output2.tif", "focus_output.tif", "Finaloutput.tif")

# Display the focus-stacked and final overlay images using SimpleITK and Matplotlib
first = sitk.ReadImage('focus_output.tif')
second = sitk.ReadImage('focus_output2.tif')
Finalphoto = sitk.ReadImage('Finaloutput.tif')

# Convert images to NumPy arrays for visualization
first_np = sitk.GetArrayFromImage(first)
second_np = sitk.GetArrayFromImage(second)
final_np = sitk.GetArrayFromImage(Finalphoto)

# Plot each image in a subplot for side-by-side comparison
plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.title("First part")
plt.imshow(first_np, cmap='gray')
plt.axis('off')  # Hide axis numbers

plt.subplot(1, 3, 2)
plt.title("Second part")
plt.imshow(second_np, cmap='gray')
plt.axis('off')  # Hide axis numbers

plt.subplot(1, 3, 3)
plt.title("Final")
plt.imshow(final_np, cmap='gray')
plt.axis('off')  # Hide axis numbers

plt.show()

