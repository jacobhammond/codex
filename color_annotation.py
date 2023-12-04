#!/usr/bin/env python3

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.segmentation import slic


# Create a class for annotations.
class Annotation:
    def __init__(self, x, y, color, hex_color):
        self.x = x
        self.y = y
        self.color = color
        self.hex_color = hex_color

# Create a list to store the annotations.
annotations = []

# Menu options
def print_menu():
    print(
        """
          Left-Click image to add annotation
          Press 'u' to remove the last annotation
          Press 'r' to reset all annotations and display original image
          Press 'e' to export annotated image file
          Press <space> to repeat this menu
          Press 'q' to quit
          """
    )

# function to find the most common colors in an image
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

# function to create a color pallete given RGB values 
def create_color_palette(common_colors):
    # create a palette image with 50 pixel high rows for each color
    palette = np.zeros((50 * len(common_colors), 100, 3), dtype=np.uint8)

    # fill each row with the color
    for i, color in enumerate(common_colors):
        palette[i * 50:(i + 1) * 50, :, :] = color

    return palette


# function to handle mouse clicks to add new annotations
def new_annotation(event, x, y, flags, param):
    global annotations, menu_select, img
    if event == cv2.EVENT_LBUTTONDOWN:
        # apply Gaussian blur to image
        blur = cv2.GaussianBlur(img, (15, 15), 0)
        # get color at position x and y in the blurred image
        color = blur[y, x]
        # convert color to HEX
        color = [int(i) for i in color]
        hex_color = '#%02x%02x%02x' % (color[2], color[1], color[0])
        new_annotation = Annotation(x, y, color, hex_color)
        # Append the new annotation to the list.
        annotations.append(new_annotation)
        # Refresh the image.
        refresh_image()
        # clear menu select so that annotation text doesn't
        # accidentally get entered as a menu option
        menu_select = ""

def remove_annotation():
    global annotations
    # check if any annotation exist to remove
    if annotations:
        print("removing last annotation")
        # remove last annotation from list
        annotations.pop()
        # Refresh the image.
        refresh_image()
    else:
        print("No annotations to remove")


def export_annotated_image():
    global img, input_image
    print("exporting annotated image...")
    # set image name to input image name without extension
    image_name = input_image.split(".")[0]
    # set annotated image name to input image name with _annotated.jpg extension
    annotated_image_name = image_name + "_annotated.jpg"
    # save annotated image
    cv2.imwrite(annotated_image_name, img)

def reset_annotations():
    global annotations
    print("resetting annotations...")
    # Clear the annotations list.
    annotations = []
    # Refresh the image.
    refresh_image()

def refresh_image():
    global img, original_image, annotations
    # Reset the image to the original image.
    img = original_image.copy()

    # Re-draw all the annotations on the image using latest annotations list
    for item in annotations:
        cv2.circle(img, (item.x, item.y), 5, (0,0,0), -1)
        cv2.circle(img, (item.x, item.y), 3, item.color, -1)
        cv2.putText(
            img,
            item.hex_color,
            (item.x + 10, item.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            item.hex_color,
            (item.x + 10, item.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            item.color,
            1,
        )
    # Refresh the image.
    cv2.imshow("Image", img)

def ui():
    global menu_select
    # If the user clicks the mouse, create a new annotation object.
    cv2.setMouseCallback("Image", new_annotation)

    # keypress variable
    menu_select = ""

    while True:
        # Wait for keypress, callback function will handle mouse click
        menu_select = cv2.waitKey(0) & 0xFF

        # If the user presses the 'u' key, remove the last annotation from the list.
        if menu_select == ord("u"):
            remove_annotation()

        # If the user presses the 'r' key, reset the annotations and display the original image.
        elif menu_select == ord("r"):
            reset_annotations()

        # If the user presses the 'e' key, export the annotated image.
        elif menu_select == ord("e"):
            export_annotated_image()

        # If the user presses the spacebar, repeat menu options
        elif menu_select == 0x20:
            print_menu()

        # If the user presses the 'q' key, quit the program.
        elif menu_select == ord("q"):
            print("Program Terminated.")
            break

    # Close all windows.
    cv2.destroyAllWindows()

def main():
    global input_image, img, original_image
    # Check if the input image and output CSV file arguments exist.
    if len(sys.argv) < 1:
        print("Error, image file arg not provided!")
        sys.exit(1)

    # otherwise load input image file
    input_image = sys.argv[1]
    img = cv2.imread(input_image)
    original_image = np.copy(img)

    # Start the window thread for the OpenCV window.
    cv2.startWindowThread()
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", img)

    # start user interface loop
    ui()


if __name__ == "__main__":
    print_menu()
    main()
