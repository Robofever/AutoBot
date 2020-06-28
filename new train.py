import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
# from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
path = os.getcwd()
clean_path = os.path.join(path,r"Clean_hsv")
unclean_path = os.path.join(path,r"Unclean_hsv")
CLASS_NAMES = np.array(['Clean','Unclean'])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,vertical_flip=True)


EPOCHS = 10
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224



# train_data_gen = image_generator.flow_from_directory(directory=str(path), \
#                                                      batch_size=BATCH_SIZE, \
#                                                      shuffle=True, \
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH), \
#                                                      classes = list(CLASS_NAMES),\
#                                                      subset='training',\
#                                                      class_mode='binary')

train_data_gen = image_generator.flow_from_directory(directory=str(path), \
                                                     batch_size=BATCH_SIZE, \
                                                     shuffle=True, \
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), \
                                                     classes = list(CLASS_NAMES),\
                                                     subset='training',\
                                                     class_mode='categorical')

from collections import Counter
counter = Counter(train_data_gen.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
print(class_weights)
# sys.exit()
STEPS_PER_EPOCH = train_data_gen.samples//BATCH_SIZE

# os.mkdir(os.path.join(path,r"Train Dataset"))
# cnt = 0
# name = os.path.join(path,r"Train Dataset")
# pbar = tqdm(total=36604)
# for x,y in train_data_gen:
#     # print(y[0])
#     for img,label in zip(x,y):
#         img=np.squeeze(img)
#         img*=255
#         img=np.array(img,dtype=np.uint8)
#         print(label)
#         # n = f"{cnt}.{label}.jpg"
#         # nme=os.path.join(name,n)
#         # cv2.imwrite(nme,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#         # if not cv2.imwrite(nme,cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
#         #     print(label,img,nme)
#         #     raise Exception("Could not write image")
#         # pbar.update()
#         # cnt+=1
#         plt.imshow(img)
#         plt.show()
#         break
#     break

# # pbar.close()
# sys.exit()

# validation_generator = image_generator.flow_from_directory(
#     directory=str(path), # same directory as training data 
#     target_size=(IMG_HEIGHT, IMG_WIDTH), \
#     batch_size=BATCH_SIZE, \
#     class_mode='binary', \
#     classes = list(CLASS_NAMES),\
#     subset='validation') # set as validation dat

validation_generator = image_generator.flow_from_directory(
    directory=str(path), # same directory as training data 
    target_size=(IMG_HEIGHT, IMG_WIDTH), \
    batch_size=BATCH_SIZE, \
    class_mode='categorical', \
    classes = list(CLASS_NAMES),\
    subset='validation')

# os.mkdir(os.path.join(path,r"Val Dataset"))
# cnt = 0
# name = os.path.join(path,r"Val Dataset")
# pbar = tqdm(total=36604)
# for x,y in validation_generator:
#     # print(y[0])
#     for img,label in zip(x,y):
#         # img=np.squeeze(img)
#         img*=255
#         img=np.array(img,dtype=np.uint8)
#         # print(y[0])
#         n = f"{cnt}.{label}.jpg"
#         nme=os.path.join(name,n)
#         cv2.imwrite(nme,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#         if not cv2.imwrite(nme,cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
#             print(label,img,nme)
#             raise Exception("Could not write image")
#         pbar.update()
#         cnt+=1
#         # plt.imshow(img)
#         # plt.show()
#         # break

# pbar.close()
# sys.exit()

# for x in validation_generator:
#     print(x[0][1])
#     plt.imshow(x[0][1])
#     plt.show()
#     break

# sys.exit()

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


# pre_trained_model = InceptionV3(input_shape = (IMG_HEIGHT, IMG_WIDTH, 3), \
#                                 include_top = False, \
#                                 weights = 'imagenet')
# pre_trained_model.summary()
# for layer in pre_trained_model.layers:
#   layer.trainable = False

# last_layer = pre_trained_model.get_layer('mixed10')
# last_output = last_layer.output
# x = pre_trained_model.output

# x = layers.Conv2D(filters=128,kernel_size=5,activation='relu')(x)
# x = layers.Conv2D(filters=64,kernel_size=5,activation='relu')(x)
# x = layers.MaxPooling2D(pool_size=(2,2))
# x = layers.Dropout(0.25)(x) 
# x = layers.Conv2D(filters=64,kernel_size=3,activation='relu')(x)
# x = layers.Conv2D(filters=32,kernel_size=3,activation='relu')(x)
# x = layers.MaxPooling2D(pool_size=(2,2))
# x = layers.Dropout(0.25)(x)
# x = layers.Flatten()
# x = layers.Dense(1024, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dropout(0.25)(x) 

# predictions = layers.Dense(1, activation='sigmoid')(x)
# predictions = layers.Dense(2, activation='softmax')(x)

from tensorflow.keras import layers,Model
############################### Model we need to play with ###############
model = tf.keras.models.Sequential()
model.add(layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Flatten())
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2,activation='softmax'))
# model = Model(inputs=pre_trained_model.input, outputs=predictions)
#############################################################################################
# from tensorflow.keras.optimizers import RMSprop

# model = keras.Sequential()
# # model.add(layers.Input())
# model.add(layers.Conv2D(32,(3,3),activation='relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Conv2D(32,(3,3),activation='relu'))
# model.add(layers.Conv2D(32,(3,3),activation='relu'))
# model.add(layers.MaxPool2D())
# model.add(layers.Conv2D(32,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(units=512,activation='relu'))
# model.add(layers.Dense(units=256,activation='relu'))
# model.add(layers.Dense(units=128,activation='relu'))
# model.add(layers.Dense(units=64,activation='relu'))
# model.add(layers.Dense(units=32,activation='relu'))
# model.add(layers.Dropout(rate=0.2))
# model.add(layers.Dense(units=1,activation='sigmoid'))


model.compile(optimizer='adam', loss = 'categorical_crossentropy', \
              metrics = ['accuracy'])
# model.compile(optimizer='adam', loss = 'categorical_crossentropy', \
#               metrics = ['accuracy'])
# categorical_crossentropy
# model.compile(optimizer=RMSprop(lr=0.001), loss = 'categorical_crossentropy', \
            #   metrics = ['accuracy'])

history = model.fit(
    train_data_gen,\
    steps_per_epoch=train_data_gen.samples//BATCH_SIZE,\
    epochs=EPOCHS,\
    validation_data = validation_generator, \
    validation_steps = validation_generator.samples // BATCH_SIZE,\
    class_weight=class_weights)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 

# acc = history.history['accuracy']
# acc = np.resize(acc,(100,1))
# plt.plot(EPOCHS,acc)
# plt.show()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()