import cv2
import numpy as np

def histogram_matching(source, template):
    """
    Match the histogram of the source image to that of the template image for RGB color channels.
    
    Parameters:
    - source: The color image to be adjusted (as a NumPy array).
    - template: The reference color image (as a NumPy array).
    
    Returns:
    - matched: The source image adjusted to match the histogram of the template.
    """
    
    matched_channels = []
    for channel in range(3):  # Process each color channel separately
        source_channel = source[:, :, channel]
        template_channel = template[:, :, channel]

        # Compute the histograms and cumulative distributions
        source_hist, _ = np.histogram(source_channel.flatten(), bins=256, range=[0, 256])
        template_hist, _ = np.histogram(template_channel.flatten(), bins=256, range=[0, 256])
        
        # Compute cumulative distribution functions (CDFs)
        source_cdf = source_hist.cumsum()
        source_cdf = (source_cdf / source_cdf[-1]) * 255  # Normalize to 255
        
        template_cdf = template_hist.cumsum()
        template_cdf = (template_cdf / template_cdf[-1]) * 255  # Normalize to 255

        # Create a lookup table to map source pixel values to template pixel values
        lookup_table = np.zeros(256)
        template_cdf = template_cdf.astype(np.uint8)
        for src_value in range(256):
            lookup_table[src_value] = np.searchsorted(template_cdf, source_cdf[src_value])

        # Apply the lookup table to get the matched channel
        matched_channel = cv2.LUT(source_channel, lookup_table.astype(np.uint8))
        matched_channels.append(matched_channel)

    # Merge the matched channels back into an RGB image
    matched = cv2.merge(matched_channels)
    return matched

# Example usage
source_image = cv2.imread('source.jpg')
template_image = cv2.imread('template.jpg')

if source_image is not None and template_image is not None:
    matched_image = histogram_matching(source_image, template_image)
    
    # Save or display the result
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error loading images. Please check file paths.")
