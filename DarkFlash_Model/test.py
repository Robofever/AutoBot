
import tensorflow
import cv2 
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import time
import picamera


def preimg(path):
    
     # import the necessary packages
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    import cv2
    #    initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    time.sleep(0.1)
    # grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    # display the image on screen and wait for a keypress
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    
    image = cv.resize(img, (224,224))
    image = np.reshape(img, [1,224, 224,1])
    print(image.shape)
    return image





model = load_model('train.h5')

#loop
img = preimg('frames/test/unclean/unclean1.mp429.jpg') #enter image path here.
prediction = np.round(model.predict(img))
print(prediction)








# img = cv.imread(r"frames/test/unclean/unclean1.mp429.jpg" ,0)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()