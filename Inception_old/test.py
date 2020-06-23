import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

from PIL import ImageGrab



# load json and create model
json_file = open('model_resnet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_resnet.h5")
# loaded_model.compile(optimizer='adam', loss = 'categorical_crossentropy', \
#               metrics = ['accuracy'])
print(loaded_model)
print("Loaded model from disk")


# cap = cv2.VideoCapture('TEST')


path = os.getcwd()
path = os.path.join(path,r"Test\Unclean2.mp4")
cap = cv2.VideoCapture(path)
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2

while(True):
    ret, frame = cap.read()
    if ret:
        resized = cv2.resize(frame, (224,224))
        resized = np.expand_dims(resized,axis=0)
        resized = np.array(resized)/255
        score = loaded_model.predict(resized)
        # print(type(score))
        # print(score)
        score = np.squeeze(score)
        print(score)
        text=f"{score}"
        # if score<0.5:
        #     text=f"Clean {score}"
        # else :
        #     text = f"Unclean {score}"
        cv2.putText(frame,text,org=(50, 50),fontFace=font,fontScale=fontScale,color=color,thickness=thickness,lineType=cv2.LINE_AA)
        cv2.imshow('frame',frame)
        cv2.imshow('new',np.squeeze(resized))
    else:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # img = ImageGrab.grab(bbox=(100,10,400,780)) #bbox specifies specific region (bbox= x,y,width,height)
    # img_np = np.array(img)
    # # cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # iiimmm = cv2.resize(img_np,(500,500))
    # resized = cv2.resize(img_np, (300,300))
    # resized = np.expand_dims(resized,axis=0)
    # resized = np.array(resized)/255
    # score = loaded_model.predict(resized)
    # # print(type(score))
    # # print(score)
    # # text=f"{score}"
    # score = np.squeeze(score)
    # print(score)
    # if score[0]>0.1:
    #     text=f"Clean {score[1]}"
    # else :
    #     text = f"Unclean {score[1]}"
    # cv2.putText(iiimmm,text,org=(50, 50),fontFace=font,fontScale=fontScale,color=color,thickness=thickness,lineType=cv2.LINE_AA)
    # cv2.imshow('frame',iiimmm)

cap.release()
cv2.destroyAllWindows()
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# resized = cv2.resize(cap1, (300,300))
# cv2.imshow('frame',cap1)
# resized = np.expand_dims(resized,axis=0)
# resized = np.array(resized)/255
# score = loaded_model.predict(resized)
# # print(type(score))
# # text=f"{score}"
# score = np.squeeze(score)
# print(score)
# # print(score.shape)
# if score[1]>0.5:
#     text=f"Clean {score[0]}"
# else :
#     text = f"Unclean {score}"
# cv2.putText(cap1,text,org=(50, 50),fontFace=font,fontScale=fontScale,color=color,thickness=thickness,lineType=cv2.LINE_AA)
