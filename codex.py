#!/usr/bin/env python3

import cv2
import sys
import os
import palette as pal
import numpy as np
import matplotlib.pyplot as plt

# MAIN PROGRAM
if __name__ == '__main__':
    # Load image
    image = cv2.imread('interior.png')

    # Extract color palette
    palette = pal.extract_color_palette(image)

    # Display image
    cv2.imshow('image', image)
    cv2.imshow('palette', palette)
    cv2.waitKey(0)