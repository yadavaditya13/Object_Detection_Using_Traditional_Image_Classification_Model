# importing all the required packages

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import ResNet50
from imutils.object_detection import non_max_suppression

from helper_mod.helper_file import sliding_window
from helper_mod.helper_file import image_pyramid

import numpy as np
import argparse
import imutils
import time
import cv2

# parsing arguments using argparse
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to Input Image...")
ap.add_argument("-s", "--size", type=str, default="(200, 150)", help="ROI size (px values)...")
ap.add_argument("-c", "--confidence", type=float, default=0.9, help="Minimum probability to filter weak detections...")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="Show extra visualization for debugging?...")

args = vars(ap.parse_args())

# initializing required variables

width = 600
pyrScale = 1.5
winStep = 16
roiSize = eval(args["size"])
inputSize = (224, 224)

# lets load our ResNet model and input image
print("[INFO] loading ResNet50 network...")
model = ResNet50(weights="imagenet", include_top=True)

# mow loading input image from disk and resizing it
print("[INFO] loading Input Image from disk...")
img = cv2.imread(args["image"])

img = imutils.resize(img, width=width)
h, w = img.shape[:2]

# lets begin the first step for object detection
# initializing image pyramid
pyramid = image_pyramid(image=img, scale=pyrScale, minSize=roiSize)

# initializing two lists to store ROIs and coordinate values of the ROIs generated
rois = []
locations = []

# lets keep a track on time taken for looping over the image
start = time.time()

# we will begin looping over each image our pyramid has produced
for image in pyramid:
    # lets determine the scale factor between original image dim and current layer of pyramid
    scale = w / float(image.shape[1])

    # for each image lets run sliding window
    for (x, y, roiImg) in sliding_window(image=image, step=winStep, ws=roiSize):
        # we will now scale the roi generated w.r.t the original image
        x = int(x * scale)
        y = int(y * scale)
        w = int(roiSize[0] * scale)
        h = int(roiSize[1] * scale)

        # pre-processing the roi
        roi = cv2.resize(roiImg, inputSize)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # appending rois and locations of image
        rois.append(roi)
        locations.append((x, y, x + w, y + h))

        # lets check if user opted for visualization
        if args["visualize"] > 0:
            # clone the original image and then draw a bounding box
            # surrounding the current region
            clone = img.copy()
            cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # lets display the visualization and current roi
            cv2.imshow("Visualization: ", clone)
            cv2.imshow("ROI: ", roiImg)
            cv2.waitKey(0)

# lets display the total time taken for this process
end = time.time()
print("[INFO] looping over Pyramid / window took: {:.5f} seconds".format(end - start))

# converting rois to numpy array
rois = np.array(rois, dtype="float32")

# passing the rois to model to make predictions
print("[INFO] classifying ROIs...")
start = time.time()
predictions = model.predict(rois)
# print("[INFO] Predictions: {}".format(predictions))
end = time.time()
print("[INFO] classifying ROIs took: {:.5f} seconds".format(end - start))

# we will now decode the predictions and initialize a dictionary which will
# map labels to ROIs associated with it
predictions = imagenet_utils.decode_predictions(predictions, top=1)
# print("[INFO] Predictions After decoding: {}".format(predictions))
labels = {}

# lets loop over predictions
for (i, p) in enumerate(predictions):
    # grab the prediction information for current roi
    (imagenetID, label, probability) = p[0]

    # filtering weak probabilities
    if probability >= args["confidence"]:
        # grabbing the location i.e. bounding box coordinates of predictions
        box = locations[i]

        # grab the list of predictions for the label and add the
        # bounding box and probability to the list
        labl = labels.get(label, [])
        labl.append((box, probability))
        labels[label] = labl

# looping over the labels for each of detected objects in the image
for label in labels.keys():
    # cloning original image
    print("[INFO] displaying results for: {}".format(label))
    clone = img.copy()

    # lets loop over the bounding boxes for current label
    for (box, probability) in labels[label]:
        # drawing bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # displaying the result before non_max_supression
        cv2.imshow("Without NMS: ", clone)
        # copying the original image
        clone = img.copy()

        # extracting the bounding boxes and associated prediction
        # probabilities, then apply non-maxima-suppression

        boxes = np.array([p[0] for p in labels[label]])
        probab = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, probab)

        # looping over the remaining boxes
        for (startX, startY, endX, endY) in boxes:
            # drawing bounding box on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # display the final output after applying NMS
    cv2.imshow("NMS: ", clone)
    cv2.waitKey(0)
# clean up
cv2.destroyAllWindows()