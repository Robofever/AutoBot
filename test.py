import tensorflow
import cv2 as cv
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

def preimg(path):
    img = cv.imread(path, 0)
    print(img.size)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7)
    img = cv.resize(img, (224,224))
    img = np.reshape(img, [1,224, 224,1])
    print(img.shape)
    return img





model = load_model('train_noob.h5')
img = preimg('clean1.mp40.jpg') #enter image path here.
model.predict(img)
