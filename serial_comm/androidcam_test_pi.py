import cv2
import numpy as np
import os
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import serial  # importing module PySerial


def preimg(path):
    img = cv2.imread(path, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 1])
    return img


if __name__ == "__main__":
    ser = serial.Serial("/dev/tty/USB0",
                        9600)  # considering USB0 to be the new port name and same baud rate and Arduino i.e., 9600
    ser.flush()
    url = "http://192.168.42.129:8080/shot.jpg"  # Enter the url from IP CAMERA here followed by /shot.jpg
    im = 'img.jpg'
    path = os.path.join(os.getcwd(), im)
    while (True):
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        cv2.imshow('pic', img)
        if cv2.waitKey(1) == 27:
            cv2.imwrite(path, img)
            break
    img = preimg(path)

    model = load_model('train_noob.h5')
    state = model.predict(img)
    print(model.predict(img))

    if state == 0:
        print("Panel is clean")
    #         while True:
    #             ser.write(b"Move forward")
    #             line0 = ser.readline().decode('utf-8').rstrip()
    #             if line0=="Done":
    #                 break

    elif state == 1:
        print("Panel is unclean")
        while True:
            ser.write(b"Do cleaning and move forward")
            line1 = ser.readline().decode('utf-8').rstrip()
            if line1 == "Cleaned":
                break