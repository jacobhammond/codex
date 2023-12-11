#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Union
import venv
import os


# Commonly used kernel sizes for general use throughout the functions
kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_9 = np.ones((9, 9), np.uint8)

# Define a class for interior reference image(s) and their color properties
class InteriorPalette:
    def __init__(
        self,
        image_file_name: str,
        image: np.ndarray,
        hsv_colors: List[List[int]],
        palette: np.ndarray
    ):
        self.image_file_name = image_file_name  # original image file name
        self.image = image  # original image
        self.hsv_colors = hsv_colors  # list of hsv colors in image
        self.palette = palette  # image of color palette

# Define a class for object(s) of interest and their respective properties
class ObjectOfInterest:
    def __init__(
        self,
        image_file_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        label: str,
        segmentation: np.ndarray,
        hsv_colors: List[List[int]],
        palette: np.ndarray,
        color_match: float
    ):
        self.image_file_name = image_file_name  # original image file name
        self.image = image  # original image
        self.mask = mask  # binary mask of object
        self.label = label  # object label (e.g. "couch") predicted by CODEX segmentation model
        self.segmentation = segmentation  # segmentation of object (for more precise color matching)
        self.hsv_colors = hsv_colors  # list of hsv colors in object
        self.palette = palette  # image of color palette
        self.color_match = color_match  # color match score of object to interior color palette

def setup():
    # check if virtual environment exists, if not create it
    if not os.path.exists('./codex-env'):
        venv.create('./codex-env', with_pip=True)
        os.system("source ./codex-env/bin/activate")
        os.system("pip install -r requirements.txt")

# Function to create a color palette of the most common colors in an image
def extract_color_palette(image):
    # resize the image to fit 200x200 pixels (for faster processing)
    image = cv2.resize(image, (200, 200))

    # apply Gaussian blur to image
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    blur = image

    # convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # segment the image into pixel regions
    segment_colors = []
    height, width, channels = image.shape
    h_step = height // 40
    w_step = width // 40
    # iterate over each region and fill with mean color to pixelate
    for i in range(0, height, h_step):
        for j in range(0, width, w_step):
            # get the region
            region = hsv[i : i + h_step, j : j + w_step]
            # get mean color of the region
            mean_color = np.mean(region, axis=(0, 1))
            segment_colors.append(mean_color)
            # fill the region with the mean color
            image[i : i + h_step, j : j + w_step] = mean_color

    # HSV values are in range 0-180 for H, 0-255 for S and V, so normalize them to 0-1
    for i in range(len(segment_colors)):
        segment_colors[i] = segment_colors[i] / [180, 255, 255]

    # Decimate the number of distinct colors in a neighborhood if euclidean distance between two colors is less than threshold
    i = 0
    while i < len(segment_colors) - 1:
        # calculate the euclidean distance between current and next color (normalized)
        dist = np.linalg.norm(segment_colors[i] - segment_colors[i + 1])
        # if the distance is less than threshold, remove the next color
        if dist < 0.15:
            segment_colors.pop(i + 1)
        else:
            i += 1

    # split the remaining hues into chunks to parse through separately
    chunks = []
    for i in range(20):
        chunks.append(
            segment_colors[
                i * len(segment_colors) // 20 : (i + 1) * len(segment_colors) // 20
            ]
        )

    for chunk in chunks:
        # sort the chunk by saturation first
        chunk.sort(key=lambda x: x[1])
        # Decimate the number of distinct saturations if euclidean distance between two colors is less than threshold
        i = 0
        while i < len(chunk) - 1:
            # calculate the euclidean distance between current and next color (normalized)
            dist = np.linalg.norm(chunk[i] - chunk[i + 1])
            # if the distance is less than threshold, remove the next color
            if dist < 0.25:
                # set the saturation of the current color to the average of the two colors
                chunk[i][1] = (chunk[i][1] + chunk[i + 1][1]) / 2
                chunk.pop(i + 1)
            else:
                i += 1
    # sort each chunk by value (brightness)
    for chunk in chunks:
        chunk.sort(key=lambda x: x[2])
        # Decimate the number of distinct values if euclidean distance between two colors is less than threshold
        i = 0
        while i < len(chunk) - 1:
            # calculate the euclidean distance between current and next color (normalized)
            dist = np.linalg.norm(chunk[i] - chunk[i + 1])
            # if the distance is less than threshold, remove the next color
            if dist < 0.25:
                # set the value of the current color to the average of the two colors
                chunk[i][2] = (chunk[i][2] + chunk[i + 1][2]) / 2
                chunk.pop(i + 1)
            else:
                i += 1

    bright = []
    muted = []
    for chunk in chunks:
        # first remove all black colors
        chunk = [color for color in chunk if color[2] > 0.1]
        for color in chunk:
            bright.append(color)

    # check in bright array if each color is similar to its neighbors with respect to hue
    bright.sort(key=lambda x: x[0])
    i = 1
    while i < len(bright) - 2:
        # calculate the euclidean distance between current and next color (normalized)
        dist = np.linalg.norm(bright[i] - bright[i - 1])
        dist2 = np.linalg.norm(bright[i] - bright[i + 1])
        dist3 = np.linalg.norm(bright[i] - bright[i + 2])
        # if the distance is less than threshold, remove the next color
        if dist < 0.25:
            bright.pop(i - 1)
        elif dist2 < 0.25:
            bright.pop(i + 1)
        elif dist3 < 0.25:
            bright.pop(i + 2)
        else:
            i += 1
    # check in bright array if each color is similar to its neighbors with respect to saturation
    bright.sort(key=lambda x: x[1])
    i = 1
    while i < len(bright) - 2:
        # calculate the euclidean distance between current and next color (normalized)
        dist = np.linalg.norm(bright[i] - bright[i - 1])
        dist2 = np.linalg.norm(bright[i] - bright[i + 1])
        dist3 = np.linalg.norm(bright[i] - bright[i + 2])
        # if the distance is less than threshold, remove the next color
        if dist < 0.25:
            bright.pop(i - 1)
        elif dist2 < 0.25:
            bright.pop(i + 1)
        elif dist3 < 0.25:
            bright.pop(i + 2)
        else:
            i += 1
    bright.sort(key=lambda x: x[0])

    # scale back up the colors to HSV range and convert to BGR
    i = 0
    color_list = []
    while i < (len(bright)):
        color =  np.uint8([(bright[i] * [180, 255, 255])])
        color_list.append(color)
        bright[i] = cv2.cvtColor(
            np.uint8([color]), cv2.COLOR_HSV2BGR
        )[0][0]
        std_dev = np.std(bright[i])
        if std_dev < 10:
            muted.append(bright[i])
            bright.pop(i)
        else:
            i += 1
    # if muted list longer than 3, shorten it to 3
    if len(muted) > 10:
        muted = muted[:10]
    # insert muted at beginning of bright
    bright = muted + bright
    
    # draw the colors into a blank image
    palette = np.zeros((50, 50 * len(bright), 3), dtype=np.uint8)
    for i, color in enumerate(bright):
        # draw  rectangle of the color
        palette[:, i * 50 : (i + 1) * 50, :] = color

    # change the color_list from a list of arrays to a list of lists
    color_list = [list(color[0]) for color in color_list]
    return palette, color_list

# Function to create a binary mask of an image
def get_obj_mask(image):
    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image to smooth the edges and reduce noise.
    blurred_image = cv2.GaussianBlur(grayscale_image, (3,3), 0)
    # Apply canny edge detection to the image.
    edges = cv2.Canny(blurred_image, 100, 200)
    # Apply thresholding to the image to create a binary mask.
    ret, mask = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    # dilate, then apply and morphological closing to the mask to fill in any holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=3)
    mask = cv2.dilate(mask, kernel_3, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=3)
    # find all closed contours in the mask and fill them in
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    # apply to original image to cut out objects from background
    mask = cv2.bitwise_and(image, image, mask=mask)
    return mask

# Function to extract the object(s) of interest from an image and apply the CODEX segmentation model
# returns a list of ObjectOfInterest instances
def isolating_seg_objects(image):
    # Load the CODEX segmentation model
    model = YOLO("yolov8n-seg.pt")
    #model = YOLO('datasets/object-training/codex.pt')

    # Run Interference and get results objects
    results = model.precict(image, imgsz=640, conf=0.5, visualize=True)



