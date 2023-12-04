#!/usr/bin/env python3

import cv2
import numpy as np

def edge_map(image):
     # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image to smooth the edges and reduce noise.
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    # Apply Laplacian edge detection to the image.
    laplace = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=3, scale=3)
    # Apply thresholding to the image to create a mask.
    ret, mask = cv2.threshold(laplace, 50, 255, cv2.THRESH_BINARY)
    # Apply a morphological closing to the mask to fill in any holes.
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Apply Laplacian edge detection to the image.
    edge_map = cv2.Laplacian(mask, cv2.CV_8U, ksize=5, scale=1)

    return edge_map

def brighten_shadows(image, edge_map):
    # dilate edge map to create a mask around potential shadow areas
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(edge_map, kernel, iterations=3)
   
    # convert the image to HSV
    hsv_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # split the image into its channels
    h, s, v = cv2.split(hsv_original)

    # check the neighborhood of pixels in the original image and get the maximum value
    # of the pixels in the neighborhood
    for i in range(8, len(hsv_original)-5, 4):
        for j in range(8, len(hsv_original[i])-5, 4):
            # get the neighborhood of the pixel
            neighborhood = hsv_original[i-4:i+5, j-4:j+5]
            # get value channel of the neighborhood
            neighborhood_v = neighborhood[:,:,2]
            # get the maximum value of the neighborhood
            max_value = np.amax(neighborhood_v)
            # set the pixel to the maximum value
            v[i-8:i][j-8:j] = max_value
                    
    # merge the channels back together
    hsv_original = cv2.merge([h, s, v])
    # convert the image back to BGR
    light_map = cv2.cvtColor(hsv_original, cv2.COLOR_HSV2BGR)
    # apply the mask to the light map to remove the shadows
    light_map = cv2.bitwise_and(light_map, light_map, mask=mask)
    # add the light map to the original image
    lightened_image = cv2.bitwise_or(image, light_map)
    
    return lightened_image
