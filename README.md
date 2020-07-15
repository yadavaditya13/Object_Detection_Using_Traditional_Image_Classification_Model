# Object_Detection_Using_Traditional_Image_Classification_Model
This programming exercise will help you understand how to implement Object Detection using Traditional Image Classification Model.

Note: 
  This program is not a very effective way to implement Object Detection but it will help you understand how it actually works step by step.
  This program takes a long time becuase of the multiple for loops used for execution, so the scripts take quite some time to run which is a lot considering the current computational world.
  
We have two scripts one for Object Detection in image called "imageClassifier_to_objectDetector.py" and the second one is for live cam called "objectDetectionLive.py".

There are comments which will help you understand each and every step in both scripts.

OverView:
  Inorder to implement Object Detection you will first take the Input image and then create an Image Pyramid, now use this pyramid full of images to pass the sliding window to get the ROIs required for detectiong objects inside the Image. 
  We will use Pre-trained ResNet50 model which is trained on Imagenet dataset with about 1000 classes. So we pass the ROIs through the moel after preprocessing and then the determined class of objects in image is then mapped with it's loaction in a dictionary.
  We will then use non-max-suppression to filter out weak detections and display the final output.

Note:
  We have a helper package created by us, which has a module with two functions one for implementing Sliding Window function and the next is for building Image Pyramid.
