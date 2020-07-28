import cv2
import numpy as np
import os
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

def preimg(path):
    img = cv2.imread(path, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    img = cv2.resize(img, (224,224))
    img = np.reshape(img, [1,224, 224,1])
    return img

if __name__ == "__main__":
    url = "http://192.168.42.129:8080/shot.jpg"                         ### Enter the url from IP CAMERA here followed by /shot.jpg 
    im = 'img.jpg'
    path = os.path.join(os.getcwd(), im)
    while(True):
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr, -1)
        cv2.imshow('pic', img)
        if cv2.waitKey(1) == 27:
            cv2.imwrite(path, img)
            break
    img = preimg(path)

    model = load_model('train_noob.h5')
    print(model.predict(img))
