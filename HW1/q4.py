# Part 1: Import Libraries and Define First Alignment and Focus Stacking Functions
import cv2
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Define the first alignment function
def align_images_first(img1, img2):
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

# Define the first focus stacking function
def focus_stack_tif_first(image1_path, image2_path, output_path="focus_stacked_output.tif"):
    img1 = tiff.imread(image1_path)
    img2 = tiff.imread(image2_path)
    
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    img1_aligned = align_images_first(img1, img2)
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



# Part 2: Define Second Alignment and Focus Stacking Functions
def align_images_second(img1, img2, use_sift=False):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if use_sift:
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    else:
        orb = cv2.ORB_create(10000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if not use_sift else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    height, width = img1.shape[:2]
    aligned_img2 = cv2.warpPerspective(img2, H, (width, height))
    
    return aligned_img2

def focus_stack_tif_second(image1_path, image2_path, output_path="focus_stacked_output2.tif"):
    img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    try:
        img2_aligned = align_images_second(img1, img2)
    except ValueError:
        print("ORB alignment failed, switching to SIFT.")
        img2_aligned = align_images_second(img1, img2, use_sift=True)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

    _, gray1_thresh = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
    _, gray2_thresh = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

    laplacian1 = cv2.Laplacian(gray1_thresh, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2_thresh, cv2.CV_64F)

    abs_laplacian1 = np.abs(laplacian1)
    abs_laplacian2 = np.abs(laplacian2)

    mask1 = abs_laplacian1 > abs_laplacian2
    mask1 = mask1.astype(np.uint8)

    mask1 = np.stack([mask1] * img1.shape[2], axis=-1) if img1.ndim == 3 else mask1

    focus_stacked = np.where(mask1 == 1, img1, img2_aligned)
    focus_stacked = cv2.cvtColor(focus_stacked, cv2.COLOR_BGR2RGB)
    tiff.imwrite(output_path, focus_stacked)
    print(f"Focus-stacked image saved as {output_path}")


# Part 3: Define Overlay Function and Execute Overlay
def overlay_images_with_mask(foreground_path, background_path, output_path="overlay_output.tif"):
    foreground = tiff.imread(foreground_path)
    background = tiff.imread(background_path)

    if foreground.shape[:2] != background.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([30, 30, 30], dtype=np.uint8)
    mask = cv2.inRange(foreground, lower_black, upper_black)

    mask_inv = cv2.bitwise_not(mask)
    foreground_no_bg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
    background_masked = cv2.bitwise_and(background, background, mask=mask)

    overlay_result = cv2.add(background_masked, foreground_no_bg)
    tiff.imwrite(output_path, overlay_result)
    print(f"Overlay image saved as {output_path}")

# Final overlay example
focus_stack_tif_first("Data/bf_7.tif", "Data/bf_5.tif", "focus_stacked_output.tif")
focus_stack_tif_second("Data/RFP_5.tif", "Data/RFP_7.tif", "focus_stacked_output2.tif")
overlay_images_with_mask("focus_stacked_output2.tif", "focus_stacked_output.tif", "overlay_output.tif")


first = sitk.ReadImage('focus_stacked_output.tif')
second = sitk.ReadImage('focus_stacked_output2.tif')
Finalphoto = sitk.ReadImage('overlay_output.tif')

first_np = sitk.GetArrayFromImage(first)
second_np = sitk.GetArrayFromImage(second)
final_np = sitk.GetArrayFromImage(Finalphoto)

plt.figure(figsize=(15, 3))

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



