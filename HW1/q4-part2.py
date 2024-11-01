import cv2
import numpy as np
import tifffile as tiff

def align_images(img1, img2, use_sift=False):
    # Convert images to grayscale for alignment
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Choose ORB or SIFT for feature detection
    if use_sift:
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    else:
        # Initialize ORB detector with more keypoints
        orb = cv2.ORB_create(10000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if not use_sift else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ensure we have enough matches
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    # Extract matched keypoints
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    # Find homography matrix
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # Warp the second image to align with the first
    height, width = img1.shape[:2]
    aligned_img2 = cv2.warpPerspective(img2, H, (width, height))
    
    return aligned_img2

def focus_stack_tif(image1_path, image2_path, output_path="focus_stacked_output.tif"):
    # Load the two .tif images using OpenCV
    img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    # Align the second image to the first, fallback to SIFT if ORB fails
    try:
        img2_aligned = align_images(img1, img2)
    except ValueError as e:
        print("ORB alignment failed, switching to SIFT.")
        img2_aligned = align_images(img1, img2, use_sift=True)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

    # Apply intensity thresholding to reduce noise in dark areas
    _, gray1_thresh = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
    _, gray2_thresh = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

    # Compute the Laplacian (sharpness) of each thresholded image
    laplacian1 = cv2.Laplacian(gray1_thresh, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2_thresh, cv2.CV_64F)

    # Calculate the absolute values (sharpness level)
    abs_laplacian1 = np.abs(laplacian1)
    abs_laplacian2 = np.abs(laplacian2)

    # Create a mask where the first image is sharper
    mask1 = abs_laplacian1 > abs_laplacian2
    mask1 = mask1.astype(np.uint8)

    # Expand the mask dimensions to match the color channels
    mask1 = np.stack([mask1] * img1.shape[2], axis=-1) if img1.ndim == 3 else mask1

    # Combine the images using the mask
    focus_stacked = np.where(mask1 == 1, img1, img2_aligned)

    # Save the final focus-stacked image as a .tif file
    # Convert from BGR to RGB for saving
    focus_stacked = cv2.cvtColor(focus_stacked, cv2.COLOR_BGR2RGB)
    tiff.imwrite(output_path, focus_stacked)
    print(f"Focus-stacked image saved as {output_path}")

# Example usage:
focus_stack_tif("Data/RFP_5.tif", "Data/RFP_7.tif", "focus_stacked_output2.tif")
