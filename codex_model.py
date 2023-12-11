# File: codex_model.py (vestigial, not used for runtime execution but provided as artifact of development)
# Author: Jacob Hammond
# Date: 12/10/2023
# Description: This file contains the code for training the CODEX model on a custom dataset

import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from ultralytics import engine
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["ULTRALYITICS_DIR"]="./datasets/object_training/"

# This function trains the CODEX model using the custom CODEX dataset on a base YOLO model trained on the COCO dataset for segmentation
def train_model():
    # check if segment training run already exists (if not, train the model)
    if not os.path.exists("./runs"):
        # Load a YOLO model to train
        model = YOLO('datasets/object_training/yolov8m-seg.pt')

        # Now train the YOLO model using the custom CODEX dataset
        results = model.train(data='datasets/object_training/data.yaml', task='segment', epochs=100, batch=128, imgsz=640, cache=True, device="cpu", workers=30)

        # Export the trained model
        model.save("datasets/object_training/codex.pt")
    else:
        # apply the custom CODEX dataset to the model weights
        model = YOLO('datasets/object_training/yolov8m-seg.pt')
        model = YOLO('datasets/object_training/codex.pt')

    # Test the model using a test codex dataset interior image
    # load the image
    #img = cv2.imread(f"datasets/examples/interior3.jpg")
    # predict the image
    #results = model(img)
    # draw the bounding boxes/segmentation
    #annotated = results[0].plot()
    # display the image
    #cv2.imshow("CODEX Model Segmentation", annotated)
    #cv2.waitKey(0)


# This function was only run at the start of the project to generate the training data for the object detection model
# it uses the object training dataset to create a list of training objects, each with an image, mask, bounding coordinates, 
# category, category index, and filename
#
# it then saves the mask and bounding coordinates to a txt file with the same name as the image file to be used as training data. 
# I started out building my dataset from scratch this way, and it turned out to be quite tedious, so I switched to using roboflow to create my dataset
# I left this function here for reference if you want to create your own dataset from scratch. A large chunk of my training images in the current
# dataset were actually created using this function. 
def create_training_obj_segments():
    class training_object:
        def __init__(self, category, category_index, filename, image, mask, bounding_coords):
            self.image = image
            self.mask = mask
            self.bounding_coords = bounding_coords
            self.category = category
            self.category_index = category_index
            self.filename = filename
    # create a list of object directories (to be used as labels too) in object training dataset
    object_dir = os.listdir("datasets/object_training/")

    # import the images in each folder into a list of images
    dataset = []
    index = 0
    for category in object_dir:
        file_list = os.listdir(f"datasets/object_training/{category}")
        #print(f"{index}: {category}", end="\n")
        for filename in file_list:
            # create training object instance
            training_object_instance = training_object(category, index, filename, cv2.imread(f"datasets/object_training/{category}/{filename}"), None, None)
            dataset.append(training_object_instance)
            
        index += 1

    # loop over each image in data 
    for training_object_instance in dataset:
        image = training_object_instance.image
        # Convert the image to grayscale.
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply canny edge detection to the image.
        edges = cv2.Canny(grayscale_image, 15, 200)
        # Apply thresholding to the image to create a binary mask.
        ret, mask = cv2.threshold(edges, 75, 255, cv2.THRESH_BINARY)
        # dilate, then apply and morphological closing to the mask to fill in any holes.
        kernel_3 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=5)
        mask = cv2.dilate(mask, kernel_3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=2)
        # find all closed contours in the mask and fill them in
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #if the countour is closed (i.e. has no parent), fill it in
        for j in range(len(hierarchy[0])):
            if hierarchy[0][j][3] == -1:
                mask = cv2.drawContours(mask, contours, j, (255, 255, 255), cv2.FILLED)
        # save mask to training object instance
        training_object_instance.mask = mask
        # get polygon coordinates of the mask
        # Extract the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Get bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Get all points in the contour
        points = np.array(largest_contour, dtype=np.int32).reshape((-1, 1, 2))
        # Create a polygon approximation of the contour
        approx_polygon = cv2.approxPolyDP(points, 0.01 * cv2.arcLength(points, True), True)
        # extract the bounding coordinates from the polygon approximation
        bounding_coords = [coord[0] for coord in approx_polygon]
        # normalize the bounding coordinates from 0-1
        bounding_coords = [[coord[0] / image.shape[1], coord[1] / image.shape[0]] for coord in bounding_coords]
        # convert the bounding coordinates to a string
        bounding_coords = [str(coord) for coord in bounding_coords]
        # join the coordinates into a single string
        bounding_coords = " ".join(bounding_coords)
        # remove the brackets from the string
        bounding_coords = bounding_coords.replace("[", "").replace("]", "")
        # remove the commas from the string
        bounding_coords = bounding_coords.replace(",", "")
        # save bounding coordinates to training object instance
        training_object_instance.bounding_coords = bounding_coords
        # join the category indext to beginning of bounding coordinates
        bounding_coords = f"{training_object_instance.category_index} {bounding_coords}"
        # create a txt file with the same name as the image file
        f = open(f"datasets/object_training/{training_object_instance.category}/{training_object_instance.filename[:-4]}.txt", "w")
        # write the bounding coordinates to the file
        f.write(bounding_coords)
        # close the file
        f.close()

    # create train and validate folders, move 80% of images to train folder and 20% to validate folder
    os.mkdir("datasets/object_training/train")
    os.mkdir("datasets/object_training/validate")
    for category in object_dir:
        file_list = os.listdir(f"datasets/object_training/{category}")
        # copy 80% of the images to train folder
        for filename in file_list[:int(len(file_list) * 0.8)]:
            os.rename(f"datasets/object_training/{category}/{filename}", f"datasets/object_training/train/{filename}")
        # copy 20% of the images to validate folder
        for filename in file_list[int(len(file_list) * 0.8):]:
            os.rename(f"datasets/object_training/{category}/{filename}", f"datasets/object_training/validate/{filename}")
    # cleanup directory and remove empty folders
    for category in object_dir:
        os.rmdir(f"datasets/object_training/{category}")

#create_training_obj_segments()
train_model()