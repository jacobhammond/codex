#!/usr/bin/env python3

import cv2
import sys
import os
import palette as pal
import masking
import numpy as np

# MAIN PROGRAM
if __name__ == '__main__':
    # Load image
    image = cv2.imread('interior.png')

    # Extract color palette
    palette, ref_colors_hsv = pal.extract_color_palette(image)
    print(ref_colors_hsv)

    # Extract object mask
    edge_map = masking.edge_map(image)
    lightened_image = masking.brighten_shadows(image, edge_map)

    # Display image
    cv2.imshow('image', image)
    cv2.imshow('lightened_image', lightened_image)
    #cv2.imshow('palette', palette)
    #cv2.imshow('edge_map', edge_map)
    cv2.waitKey(0)