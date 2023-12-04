#!/usr/bin/env python3

import cv2
import numpy as np

# function to create a color palette of the most common colors in an image
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
    while i < (len(bright)):
        bright[i] = cv2.cvtColor(
            np.uint8([[(bright[i] * [180, 255, 255])]]), cv2.COLOR_HSV2BGR
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

    return palette


# MAIN PROGRAM
if __name__ == "__main__":
    # Load image
    image = cv2.imread("interior.jpg")

    # Extract color palette
    palette = extract_color_palette(image)
