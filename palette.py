import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic


def extract_common_colors(image):
 # Convert image to LAB color space
    image_lab = rgb2lab(image)

    # Perform SLIC (Superpixels) segmentation
    segments = slic(image_lab, n_segments=10, compactness=4.8)

    # Extract color centroids from superpixels
    unique_labels = np.unique(segments)
    color_centroids = []

    for label in unique_labels:
        segment_pixels = image_lab[segments == label]
        mean_color = np.mean(segment_pixels, axis=0)
        color_centroids.append(mean_color)

    color_centroids = np.array(color_centroids)

    # Convert LAB colors to RGB
    common_colors = rgb2lab(color_centroids)

    # Return common colors in RGB format
    return common_colors


# Load image
image = cv2.imread('interior.png')

# Extract common colors using the simplified method
common_colors = extract_common_colors(image)

# Display common colors
palette = np.zeros((50 * len(common_colors), 100, 3), dtype=np.uint8)
for i, color in enumerate(common_colors):
    palette[i * 50:(i + 1) * 50, :, :] = color

# Display palette
cv2.imshow('Color Palette', palette)
# show the input image
cv2.imshow("Input Image", image)
cv2.waitKey(0)