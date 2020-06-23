import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join , abspath
import cv2
from tensorflow import keras
#import pathlib
#import PIL
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model


path = os.getcwd()
CLASS_NAMES = np.array(['Clean','Unclean'])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.15,horizontal_flip=True,brightness_range=(0.8,1))
EPOCHS = 10
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


# sys.exit()


train_data_gen = image_generator.flow_from_directory(directory=str(path), \
                                                     batch_size=BATCH_SIZE, \
                                                     shuffle=True, \
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), \
                                                     classes = list(CLASS_NAMES),\
                                                     subset='training',\
                                                     class_mode='categorical')


STEPS_PER_EPOCH = train_data_gen.samples//BATCH_SIZE
from collections import Counter
counter = Counter(train_data_gen.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
print(class_weights)

validation_generator = image_generator.flow_from_directory(
    directory=str(path), # same directory as training data 
    target_size=(IMG_HEIGHT, IMG_WIDTH), \
    batch_size=BATCH_SIZE, \
    class_mode='categorical', \
    classes = list(CLASS_NAMES),\
    subset='validation') # set as validation data
    
pre_trained_model = tf.keras.applications.resnet50.ResNet50(
   input_shape = (IMG_HEIGHT, IMG_WIDTH,3), \
   include_top = False, \
   weights = 'imagenet')
   
# pre_trained_model.summary()
for layer in pre_trained_model.layers:
    layer.trainable = False
   
# pre_trained_model.summary()

# last_layer = pre_trained_model.get_layer('add_142')
last_output = pre_trained_model.output
print(last_output)
x = pre_trained_model.output
x = layers.Flatten()(last_output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x) 

predictions = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=pre_trained_model.input, outputs=predictions)

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
    train_data_gen, \
    steps_per_epoch=train_data_gen.samples//BATCH_SIZE, \
    epochs=EPOCHS, \
    validation_data = validation_generator,  \
    validation_steps = validation_generator.samples // BATCH_SIZE, \
    class_weight=class_weights)

# serialize model to JSON
model_json = model.to_json()
with open("model_resnet.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_resnet.h5")
print("Saved model to disk")
 
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