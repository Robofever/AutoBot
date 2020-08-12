import cv2
import numpy as np
import os
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import serial  # importing module PySerial
import time


def preimg(path):
    img = cv2.imread(path, 0)
    img = cv2.adaptiveThreshold(img, 255, 

cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 1])
    return img


if __name__ == "__main__":
    ser = serial.Serial("/dev/ttyACM0",
                        9600)  # considering USB0 to be the 

new port name and same baud rate and Arduino i.e., 9600
    ser.flush()
    url = "http://192.168.2.4:8080/shot.jpg"  # Enter the url 

from IP CAMERA here followed by /shot.jpg
    im = 'img.jpg'
    path = os.path.join(os.getcwd(), im)
#     while (True):
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), 

# dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         cv2.imshow('pic', img)
#         if cv2.waitKey(1) == 27:
#             cv2.imwrite(path, img)
#             break
    # img = preimg(path)

    model = load_model('train_all_data.h5')
    state = np.argmax(model.predict(img))
    print(model.predict(img))

    
    bool isUnclean = False
    int a=0
    int b=0
   
    while (True):
        # ser.write(b"Move forward")
     
        if isUnclean == True :
            a += 1
            b=0
        
        if isUnclean == False: 
            a = 0
            b += 1
             

        line0 = ser.readline().decode('utf-8').rstrip()
        if line0=="-1":
            img = preimg(path)
            state = np.argmax(np.round(model.predict(img)))
            if state == 0 :  
                serial.write(b"0") #Clean
                # serial.write(b"Move Forward")
                
                isUnclean = False
                if b == 2:
                    time.sleep(43200) #12hr CoolDown
               
            
            if state == 1 :
                serial.write(b"1") #Unclean
                isUnclean = True
                if a == 2:
                    print("Problem with Cleaning")
                    time.sleep(43200) #12hr CoolDown
               
                    
                # serial.write(b"Start Cleaning")
                continue

    