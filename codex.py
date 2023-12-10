#!/usr/bin/env python3

import sys
import cv2
import os
import codex_common as codex

# MAIN PROGRAM
if __name__ == '__main__':

    # Get a list of interior reference images
    interior_ref_files = os.listdir("codex-dataset/interiors/")

    # import the interior reference images into class and get their color palette
    for filename in interior_ref_files:
        image = cv2.imread("codex-dataset/interiors/" + filename)
        # call the extract_color_palette function 
        palette, hsv_colors = codex.extract_color_palette(image)
        # save the color palette to a file
        #cv2.imwrite("codex-dataset/interiors/" + filename + "_palette.jpg", palette)
        # create an instance of the InteriorPalette class
        codex.InteriorPalette(filename, image, hsv_colors, palette)

    # Extract objects
    objects = codex.get_obj_mask(image)

    # Display image
    #cv2.imshow('image', image)
    cv2.imshow('objects', objects)
    #cv2.imshow('palette', palette)
    cv2.waitKey(0)