# importing required packages

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import  imagenet_utils
from tensorflow.keras.applications import ResNet50

from helper_mod.helper_file import image_pyramid, sliding_window
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream

import numpy as np
import argparse
import imutils
import time
import cv2

# parsing arguments
ap = argparse.ArgumentParser()

ap.add_argument("-s", "--size", type=str, default="(200, 150)", help="ROI size (px values)...")
ap.add_argument("-c", "--confidence", type=float, default=0.9, help="Minimum confidence value for filtering...")

args = vars(ap.parse_args())

# initializing required variables

width = 600
pyrScale = 1.5
winStep = 16
roiSize = eval(args["size"])
inputSize = (224, 224)

# loading our pre-trained model
print("[INFO] loading ResNet50 model...")
model = ResNet50(weights="imagenet", include_top=True)

# initializing video stream
print("[INFO] We are going Live...")
vs = VideoStream(src=0).start()
time.sleep(1)

# looping over frames one by one
while True:
    # reading the current frame
    frame = vs.read()
    frame = imutils.resize(frame, width=width)
    h, w = frame.shape[:2]
    origFrame = frame.copy()

    # lets begin the first step for object detection
    # initializing image pyramid
    pyramid = image_pyramid(image=frame, scale=pyrScale, minSize=roiSize)

    # initializing two lists to store ROIs and coordinate values of the ROIs generated
    rois = []
    locations = []

    # we will begin looping over each image our pyramid has produced
    for image in pyramid:
        # lets determine the scale factor between original image dim and current layer of pyramid
        scale = w / float(image.shape[1])

        # for each image lets run sliding window
        for (x, y, roiImg) in sliding_window(image, winStep, roiSize):
            # we will now scale the roi generated w.r.t the original image
            x = int(x * scale)
            y = int(y * scale)
            w = int(roiSize[0] * scale)
            h = int(roiSize[1] * scale)

            # pre-processing the roi
            #roiImg = cv2.imread(roiImg)
            roi = cv2.resize(roiImg, inputSize)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)

            # appending rois and locations of image
            rois.append(roi)
            locations.append((x, y, x+w, y+h))

    # converting rois to numpy array
    rois = np.array(rois, dtype="float32")
    # passing the rois to model to make predictions
    predictions = model.predict(rois)

    # we will now decode the predictions and initialize a dictionary which will
    # map labels to ROIs associated with it
    predictions = imagenet_utils.decode_predictions(predictions, top=1)
    labels = {}

    # lets loop over predictions
    # here we have mapped label ==> (box, probability)
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

    # looping over the labels for each of detected objects in the frames
    for label in labels.keys():

        # lets loop over the bounding boxes for current label
        for (box, probability) in labels[label]:
            # grabbing box dimensions
            (startX, startY, endX, endY) = box

            # cloning original image
            clone = origFrame.copy()

            # extracting the bounding boxes and associated prediction
            # probabilities, then apply non-maxima-suppression

            boxes = np.array([p[0] for p in labels[label]])
            probab = np.array([p[1] for p in labels[label]])
            boxes = non_max_suppression(boxes, probab)

            # looping over the boxes
            for (startX, startY, endX, endY) in boxes:
                # drawing bounding box on the frames
                cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # displaying the frame
        cv2.imshow("Frame: ", clone)

    # exit path from stream
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# clean up task
cv2.destroyAllWindows()
vs.stop()