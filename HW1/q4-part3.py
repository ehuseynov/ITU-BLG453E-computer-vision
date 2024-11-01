import numpy as np
import cv2
import tifffile as tiff

def overlay_images_with_mask(foreground_path, background_path, output_path="overlay_output.tif"):
    # Load both .tif images using tifffile to preserve original color format
    foreground = tiff.imread(foreground_path)
    background = tiff.imread(background_path)

    # Ensure both images are the same size by resizing the foreground to match background dimensions
    if foreground.shape[:2] != background.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # Create a mask to detect the black background in the foreground image
    # Adjust the threshold as needed to remove near-black areas
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([30, 30, 30], dtype=np.uint8)
    mask = cv2.inRange(foreground, lower_black, upper_black)

    # Invert the mask to keep only the non-black parts of the foreground
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverted mask to isolate the non-black parts of the foreground
    foreground_no_bg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)

    # Use the mask to clear the area on the background where the foreground will be overlaid
    background_masked = cv2.bitwise_and(background, background, mask=mask)

    # Combine the foreground (without black background) onto the background
    overlay_result = cv2.add(background_masked, foreground_no_bg)

    # Save the final overlaid image as a .tif file
    tiff.imwrite(output_path, overlay_result)
    print(f"Overlay image saved as {output_path}")

# Example usage:
overlay_images_with_mask("focus_stacked_output2.tif", "focus_stacked_output.tif", "overlay_output.tif")
