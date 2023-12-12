#!/usr/bin/env python3
# File: codex_common.py 
# Author: Jacob Hammond
# Date: 12/10/2023
# Description: This file contains common functions and classes used by codex.py and codex_model.py

import sys
import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from typing import List, Union
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["ULTRALYITICS_DIR"]="./datasets/object_training/"

        

# Commonly used kernel sizes for general use throughout the functions
kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_9 = np.ones((9, 9), np.uint8)
categories = ['bed', 'chair', 'clock', 'couch', 'curtain', 'dresser', 'lamp', 'light', 
              'pillow', 'plant', 'rug', 'shelf', 'stool', 'table', 'vase', 'wall-art']
neutral_colors_hsv = [
    # Grays
    (0, 0, 1.0),  # White
    (0, 0, 0.501963137255),  # Light gray
    (0, 0, 0.25095686274),   # Gray
    (0, 0, 0.12549043137),   # Dark gray
    (0, 0, 0.0),    # Black

    # Browns
    (0.0416666, 0.2, 0.39215), # Light brown
    (0.055555, 0.25, 0.3137254),  # Brown
    (0.069444, 0.3, 0.23529411),  # Dark brown

    # Beiges
    (0.08333333, 0.1, 0.3921568803), # Light beige
    (0.09722222, 0.15, 0.31372784),  # Beige
    (0.11111111, 0.2, 0.23525882),  # Dark beige

    # Creams
    (0.125, 0.05, 0.39215509803),  # Light cream
    (0.138889, 0.1, 0.31396784),  # Cream
    (0.1527778, 0.15, 0.23529882),  # Dark cream
]

# Define a class for interior reference image(s) and their color properties
class ReferencePalette:
    def __init__(
        self,
        image_file_name: str,
        image: np.ndarray,
        hsv_colors: List[np.ndarray],
        palette: np.ndarray
    ):
        self.image_file_name = image_file_name  # original image file name
        self.image = image  # original image
        self.hsv_colors = hsv_colors  # list of hsv colors in image
        self.palette = palette  # image of color palette

# Define a class for segmented object(s) and their color properties
class SegmentPalette:
    def __init__(
        self,
        image: np.ndarray,
        crop: np.ndarray,
        mask: np.ndarray,
        label: str,
        conf: float,
        hsv_colors: List[np.ndarray],
        palette: np.ndarray,
        color_match: str
    ):
        self.image = image # extracted segmented object image
        self.crop = crop # cropped image of object
        self.mask = mask # mask of object
        self.label = label # label of object in CODEX category
        self.conf = conf # confidence of object detection
        self.hsv_colors = hsv_colors # list of hsv colors in object
        self.palette = palette # image of color palette
        self.color_match = color_match # color match score of object to interior color palette


# Function to create a color palette of the most common colors in an image
def extract_color_palette(image):
    # resize the image to fit 200x200 pixels (for faster processing)
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)

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
        if dist < 0.10:
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
            if dist < 0.15:
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
            if dist < 0.15:
                # set the value of the current color to the average of the two colors
                chunk[i][2] = (chunk[i][2] + chunk[i + 1][2]) / 2
                chunk.pop(i + 1)
            else:
                i += 1

    bright = []
    muted = []
    for chunk in chunks:
        # first remove all black colors or white colors
        chunk = [color for color in chunk if (color[2] > 0.1 and color[2] < 0.95)]
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
        if dist < 0.15:
            bright.pop(i - 1)
        elif dist2 < 0.15:
            bright.pop(i + 1)
        elif dist3 < 0.15:
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
        if dist < 0.15:
            bright.pop(i - 1)
        elif dist2 < 0.15:
            bright.pop(i + 1)
        elif dist3 < 0.15:
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

    # randomly remove indexes from bright until length is 10
    while len(bright) > 20:
        bright.pop(np.random.randint(0, len(bright)))
    
    # draw the colors into a blank image
    palette = np.zeros((50, 50 * len(bright), 3), dtype=np.uint8)
    for i, color in enumerate(bright):
        # draw  rectangle of the color
        palette[:, i * 50 : (i + 1) * 50, :] = color
    # change the color_list from a list of arrays to a list of lists
    #color_list = [list(color[0]) for color in color_list]
    return palette, color_list

# function for smaller scale color palette extraction for segmented objects
# fine tuned for smaller images that have been segmented and modelled to 
# ignore binary mask and only extract colors from the object(s) of interest
def extract_object_color_palette(image, mask):
    # apply mask to image
    img = cv2.bitwise_and(image, image, mask=mask)
     # convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split image into 32x32 pixel regions
    segment_colors = []
    height, width, channels = img.shape
    h_step = height // 32
    w_step = width // 32
    # iterate over each region and fill with mean color to pixelate
    for i in range(0, height, h_step):
        for j in range(0, width, w_step):
            # get the region
            region = hsv[i : i + h_step, j : j + w_step]
            # get mean color of the region
            mean_color = np.mean(region, axis=(0, 1))
            segment_colors.append(mean_color)
            # fill the region with the mean color
            img[i : i + h_step, j : j + w_step] = mean_color
    # HSV values are in range 0-180 for H, 0-255 for S and V, so normalize them to 0-1
    for i in range(len(segment_colors)):
        segment_colors[i] = segment_colors[i] / [180, 255, 255]

    # first remove all totally black colors to ignore the binary mask
    bright = []
    chunk = [color for color in segment_colors if color[2] > 0.1]

    # Decimate the number of distinct colors in a neighborhood if euclidean distance between two colors is less than threshold
    i = 0
    while i < len(segment_colors) - 1:
        # calculate the euclidean distance between current and next color (normalized)
        dist = np.linalg.norm(segment_colors[i] - segment_colors[i + 1])
        # if the distance is less than threshold, remove the next color
        if dist < 0.30:
            segment_colors.pop(i + 1)
        else:
            i += 1
    
    for color in chunk:
        bright.append(color)

    # scale back up the colors to HSV range and convert to BGR
    i = 0
    color_list = []
    muted = []
    while i < (len(bright)):
        color =  np.uint8([(bright[i] * [180, 255, 255])])
        color_list.append(color)
        bright[i] = cv2.cvtColor(
            np.uint8([color]), cv2.COLOR_HSV2BGR
        )[0][0]
        std_dev = np.std(bright[i])
        if std_dev < 20:
            muted.append(bright[i])
            bright.pop(i)
        else:
            i += 1
    # if muted list longer than 3, shorten it to 3
    if len(muted) > 10:
        muted = muted[:10]
    # insert muted at beginning of bright
    bright = muted + bright

    # randomly remove indexes from bright until length is 10
    while len(bright) > 10:
        bright.pop(np.random.randint(0, len(bright)))
    
    # draw the colors into a blank image
    palette = np.zeros((50, 50 * len(bright), 3), dtype=np.uint8)
    for i, color in enumerate(bright):
        # draw  rectangle of the color
        palette[:, i * 50 : (i + 1) * 50, :] = color
    # change the color_list from a list of arrays to a list of lists
    #color_list = [list(color[0]) for color in color_list]
    return palette, color_list


# Function to compare two color palettes and return a score of how similar they are
def color_score(ref_hsv_colors, obj_hsv_colors):
    relative_score = 0
    neutral_score = 0
    # copy a list of colors that are in both palettes
    ref_colors = ref_hsv_colors
    obj_colors = obj_hsv_colors
    # get HSV histograms of each list
    ref_hist = cv2.calcHist([np.uint8(ref_colors)], [0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256])
    obj_hist = cv2.calcHist([np.uint8(obj_colors)], [0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256])
    # HSV values are in range 0-180 for H, 0-255 for S and V, so normalize them to 0-1
    for i in range(len(ref_colors)):
        ref_colors[i] = ref_colors[i] / [180, 255, 255]
    for i in range(len(obj_colors)):
        obj_colors[i] = obj_colors[i] / [180, 255, 255]

    # compare to neutral tones white, black, gray, brown, beige, and cream
    for neutral_color in neutral_colors_hsv:
        for color in obj_colors:
            dist = np.linalg.norm(neutral_color - color)
            if dist < 0.15:
                neutral_score += 1
        
    # compare relative to each other
    for r_color in ref_colors:
        # compare to ref_hsv_color by euclidean distance
        for o_color in obj_colors:
            # calculate the euclidean distance between current and next color (normalized)
            dist = np.linalg.norm(r_color - o_color)
            # shot distance indicates a good match, so add to score
            if dist < 0.15:
                relative_score += 1
    
    if relative_score > 600:
        match_str = "Good"
    else:
        match_str = "Poor"
           
    return match_str

# Function to resize images to target size, or to fit in standard viewing window and keep aspect ratio
def resize_image(image, target=None):

    # Define the default target width and height
    default_width = 640
    default_height = 480

    if target is not None:
        # Calculate the target width and height based on the provided target image
        target_width = target.shape[1]
        target_height = target.shape[0]
    else:
        # Use the default target width and height
        target_width = default_width
        target_height = default_height
    
    # Check if the target is smaller than the image
    if target_width < image.shape[1] or target_height < image.shape[0]:

        # Calculate the original aspect ratio
        original_aspect_ratio = image.shape[1] / image.shape[0]

        # Determine the scaling factor based on the target dimensions
        if target_width / target_height > original_aspect_ratio:
            # Scale based on width
            scaling_factor = target_width / image.shape[1]
            new_width = target_width
            new_height = int(image.shape[0] * scaling_factor)
        else:
            # Scale based on height
            scaling_factor = target_height / image.shape[0]
            new_width = int(image.shape[1] * scaling_factor)
            new_height = target_height

        # Resize the image
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    else : # target is larger, pad edges with black the difference
        # Get the original image dimensions
        original_width = image.shape[1]
        original_height = image.shape[0]

        # Calculate the padding size
        padding_width = target_width - original_width
        padding_height = target_height - original_height

        # Create a new black image with the target size
        padded_image = cv2.copyMakeBorder(
            image,
            top=int(padding_height / 2),  # Top padding
            bottom=int(padding_height / 2),  # Bottom padding
            left=int(padding_width / 2),  # Left padding
            right=int(padding_width / 2),  # Right padding
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),  # Black color
        )
    return padded_image

# Function to create binary mask based on contours (deprecated, replaced by isolating_seg_objects)
def get_bounding_contour(image):
    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image to smooth the edges and reduce noise.
    blurred_image = cv2.GaussianBlur(grayscale_image, (3,3), 0)
    # Apply canny edge detection to the image.
    edges = cv2.Canny(blurred_image, 100, 200)
    # Apply thresholding to the image to create a binary mask.
    ret, mask = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    # dilate, then apply and morphological closing to the mask to fill in any holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=1)
    # find all closed contours in the mask and fill them in
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    # apply to original image to cut out objects from background
    mask = cv2.bitwise_and(image, image, mask=mask)
    return mask

# Function to extract the object(s) of interest from an image and apply the CODEX segmentation model
# returns a list of ObjectOfInterest instances
def isolating_seg_objects(image):
    image = np.copy(image)
    # Load the CODEX segmentation model
    model = YOLO("yolov8m-seg.pt")
    model = YOLO('codex.pt')

    # Run Interference and get results objects
    results = model.predict(image, imgsz=640, max_det=9)

    # iterate over results
    isolated_objects = []
    for result in results:
        i = 1
        # iterate each object contour (if more than one detection by model)
        for ci, c in enumerate(result):
            # get detection class
            label = c.names[c.boxes.cls.tolist().pop()]
            # get confidence of top 1
            conf = c.boxes.conf.tolist().pop()
            # get a binary mask
            mask = np.zeros(image.shape[:2], np.uint8)
            # extract contour result
            countour = c.masks.xy.pop()
            # change the type to np.int32
            countour = countour.astype(np.int32)
            # reshape the contour to be 2D
            countour = countour.reshape((-1, 1, 2))
            # draw the contour onto mask
            _ = cv2.drawContours(mask, [countour], -1, (255, 255, 255), cv2.FILLED)
            # ensure the mask has the same size and data type as the image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            b_mask = mask.astype(image.dtype)
            # isolate the object with transparent background
            #seg_image = np.dstack([image, mask])
            # isolate the object with black background
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            iso_full = cv2.bitwise_and(mask3ch, image)
            #  Bounding box coordinates
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            # Crop image to object region
            iso_crop = iso_full[y1:y2, x1:x2]
            # append to isolated objects list
            isolated_objects.append([label, conf, iso_full, iso_crop, b_mask])

    return isolated_objects

# function to call the extract_color_palette function for a list segmented object(s)
# and perform color matching for each segmented object versus the ReferencePalette instance
# if the object is not a CODEX category, it is skipped. Otherwise the color matching is performed
# and a SegmentPalette instance is initialized and appended to a list of results
# output is a list of SegmentPalette instances with color matching results
def color_match(ref, segmented_objects):
    results = []
    for object in segmented_objects:
        # get the label and only compare if the detection is a CODEX category
        label = object[0]
        conf = object[1]
        
        if label not in categories:
            continue
        # get the segment image
        iso_full = object[2]
        iso_crop = object[3]
        mask = object[4]
        # extract the color palette for each segment
        seg_palette, seg_hsv_colors = extract_object_color_palette(iso_full, mask)
        # call the color_match function
        color_match = color_score(ref.hsv_colors, seg_hsv_colors)
        # initialize an instance of the SegmentPalette class and append to results list
        results.append(SegmentPalette(iso_full, iso_crop, mask, label, conf, seg_hsv_colors, seg_palette, color_match))

    return results

# function to create a mosaic of the segmented objects for easy viewing
def create_image_mosaic(images):
    # if the number of images is odd, add a blank image to the end
    if len(images) % 2 != 0:
        images.append(np.zeros(images[0].shape, dtype=np.uint8))
    print(len(images))
    columns = []
    result = []
    i=0
    while i < (len(images)-1):
        if len(columns) == 3:
            result.append(cv2.vconcat([columns[0], columns[1], columns[2]]))
            columns = []
        columns.append(images[i])
        i += 1
    
    remainder = 3 - len(columns)

    for i in range (remainder):
        columns.append(np.zeros(images[0].shape, dtype=np.uint8))
    
    result.append(cv2.vconcat([columns[0], columns[1], columns[2]]))

    combined_image = result[0]    
    
    if len(result) > 1:
        for i in range(1, len(result)):
            combined_image = cv2.hconcat([combined_image, result[i]])

    return combined_image

# Function to display the resulting output of the CODEX program
# input is a ReferencePalette instance and a list of SegmentPalette instances to compare to
def display_results(ref, seg_results):

    # resize the height of the ref palette to the height of the reference image
    ref.palette = cv2.resize(ref.palette, (ref.image.shape[1], ref.palette.shape[0]))
    # add the ref palette to the right side of the reference image
    ref_result = np.vstack((ref.image, ref.palette))
    ref_result = resize_image(ref_result)

    segment_display_images= []
    # iterate over each segment result
    for seg in seg_results:
        # add the label to the result image
        cv2.putText(seg.image, f"category: {seg.label} ({int(seg.conf*100)}% conf)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        # add the color match to the blank image
        cv2.putText(seg.image, f"color_match: {seg.color_match}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        # resize the height of color palette to match the blank result image
        seg.palette = cv2.resize(seg.palette, (seg.image.shape[1], seg.palette.shape[0]))
        # add the segment color palette to the bottom side of the blank image
        seg_display = np.vstack((seg.image, seg.palette))
        # append the segment display image to the list of segment display images
        segment_display_images.append(seg_display)

    # resize segment image(s)
    target = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(len(segment_display_images)):
        segment_display_images[i] = resize_image(segment_display_images[i], target)

    # create mosaic of segment display images
    if len(segment_display_images) > 1:
        result = create_image_mosaic(segment_display_images)
            
    # otherwise just set the result to the first segment display image
    else:
        result = resize_image(segment_display_images[0], target)
    cv2.imshow("CODEX Model Segmentation", result)
    cv2.imshow("Reference Image", ref_result)
    cv2.waitKey(0)
