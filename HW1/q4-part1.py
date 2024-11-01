import cv2
import numpy as np
import tifffile as tiff

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (width, height))
    return aligned_img1

def focus_stack_tif(image1_path, image2_path, output_path="focus_stacked_output.tif"):
    img1 = tiff.imread(image1_path)
    img2 = tiff.imread(image2_path)
    
    # Resize img1 to match img2 if their shapes are different
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    img1_aligned = align_images(img1, img2)
    gray1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    laplacian1 = cv2.Laplacian(gray1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2, cv2.CV_64F)
    abs_laplacian1 = np.abs(laplacian1)
    abs_laplacian2 = np.abs(laplacian2)
    mask1 = abs_laplacian1 > abs_laplacian2
    mask1 = mask1.astype(np.uint8)
    mask1 = np.stack([mask1] * img1.shape[2], axis=-1) if img1.ndim == 3 else mask1

    focus_stacked = np.where(mask1 == 1, img1_aligned, img2)
    tiff.imwrite(output_path, focus_stacked)
    print(f"Focus-stacked image saved as {output_path}")

# Example usage:
focus_stack_tif("Data/bf_7.tif", "Data/bf_5.tif", "focus_stacked_output.tif")



