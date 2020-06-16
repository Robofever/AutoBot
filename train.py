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
clean_path = os.path.join(path,r"Clean")
unclean_path = os.path.join(path,r"Unclean")
CLASS_NAMES = np.array(['Clean','Unclean'])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.15,horizontal_flip=True,vertical_flip=True,brightness_range=(0.7,1))


EPOCHS = 30
BATCH_SIZE = 32
IMG_HEIGHT = 150*2
IMG_WIDTH = 150*2



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

pre_trained_model = InceptionV3(input_shape = (IMG_HEIGHT, IMG_WIDTH, 3), \
                                include_top = False, \
                                weights = 'imagenet')
# pre_trained_model.summary()
for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed10')
last_output = last_layer.output
x = pre_trained_model.output
x = layers.Flatten()(last_output)
# x = layers.Dense(1024, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x) 

# predictions = layers.Dense(1, activation='sigmoid')(x)
predictions = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=pre_trained_model.input, outputs=predictions)

from tensorflow.keras.optimizers import RMSprop

# model.compile(optimizer=RMSprop(lr=0.001), loss = 'binary_crossentropy', \
#               metrics = ['accuracy'])
model.compile(optimizer='adam', loss = 'categorical_crossentropy', \
              metrics = ['accuracy'])
# categorical_crossentropy
# model.compile(optimizer=RMSprop(lr=0.001), loss = 'categorical_crossentropy', \
            #   metrics = ['accuracy'])

history = model.fit(
    train_data_gen,\
    steps_per_epoch=train_data_gen.samples//BATCH_SIZE,\
    epochs=EPOCHS,\
    validation_data = validation_generator, \
    validation_steps = validation_generator.samples // BATCH_SIZE,\
    class_weight=class_weights,\
    verbose=1)

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
############# Last layer sigmoid units=1 binary  rmsprop 0.001 ##############################
# 1215/1215 [==============================] - 110s 91ms/step - loss: 0.6750 - accuracy: 0.9095 - val_loss: 0.1739 - val_accuracy: 0.9578
# Epoch 2/100
# 1215/1215 [==============================] - 110s 90ms/step - loss: 0.2476 - accuracy: 0.9528 - val_loss: 0.2367 - val_accuracy: 0.9498
# Epoch 3/100
# 1215/1215 [==============================] - 115s 94ms/step - loss: 0.2083 - accuracy: 0.9607 - val_loss: 0.2878 - val_accuracy: 0.9603
# Epoch 4/100
# 1215/1215 [==============================] - 110s 91ms/step - loss: 0.1848 - accuracy: 0.9652 - val_loss: 0.4660 - val_accuracy: 0.9603
# Epoch 5/100
# 1215/1215 [==============================] - 106s 87ms/step - loss: 0.2005 - accuracy: 0.9662 - val_loss: 0.4164 - val_accuracy: 0.9581
# Epoch 6/100
# 1215/1215 [==============================] - 110s 91ms/step - loss: 0.1961 - accuracy: 0.9684 - val_loss: 0.3838 - val_accuracy: 0.9623
# Epoch 7/100
# 1215/1215 [==============================] - 108s 89ms/step - loss: 0.1700 - accuracy: 0.9703 - val_loss: 0.5388 - val_accuracy: 0.9679
# Epoch 8/100
# 1215/1215 [==============================] - 114s 94ms/step - loss: 0.1816 - accuracy: 0.9710 - val_loss: 0.5923 - val_accuracy: 0.9540
# Epoch 9/100
# 1215/1215 [==============================] - 116s 95ms/step - loss: 0.1691 - accuracy: 0.9710 - val_loss: 0.6656 - val_accuracy: 0.9308
# Epoch 10/100
# 1215/1215 [==============================] - 116s 95ms/step - loss: 0.2096 - accuracy: 0.9723 - val_loss: 0.5521 - val_accuracy: 0.9522
# Epoch 11/100
# 1215/1215 [==============================] - 112s 93ms/step - loss: 0.1580 - accuracy: 0.9720 - val_loss: 0.6885 - val_accuracy: 0.9448
# Epoch 12/100
# 1215/1215 [==============================] - 112s 93ms/step - loss: 0.1681 - accuracy: 0.9727 - val_loss: 0.5226 - val_accuracy: 0.9525
# Epoch 13/100
# 1215/1215 [==============================] - 112s 92ms/step - loss: 0.1830 - accuracy: 0.9711 - val_loss: 0.6726 - val_accuracy: 0.9530
# Epoch 14/100
# 1215/1215 [==============================] - 110s 91ms/step - loss: 0.1476 - accuracy: 0.9735 - val_loss: 0.9158 - val_accuracy: 0.9597
# Epoch 15/100
# 1215/1215 [==============================] - 112s 92ms/step - loss: 0.1457 - accuracy: 0.9733 - val_loss: 0.7829 - val_accuracy: 0.9388
# Epoch 16/100
# 1215/1215 [==============================] - 137s 113ms/step - loss: 0.1425 - accuracy: 0.9737 - val_loss: 0.6660 - val_accuracy: 0.9433
# Epoch 17/100
# 1215/1215 [==============================] - 118s 97ms/step - loss: 0.1544 - accuracy: 0.9750 - val_loss: 0.7195 - val_accuracy: 0.9556
# Epoch 18/100
# 1215/1215 [==============================] - 106s 87ms/step - loss: 0.1491 - accuracy: 0.9739 - val_loss: 0.6444 - val_accuracy: 0.9560
# Epoch 19/100
# 1215/1215 [==============================] - 105s 87ms/step - loss: 0.1261 - accuracy: 0.9753 - val_loss: 0.7027 - val_accuracy: 0.9366
# Epoch 20/100
# 1215/1215 [==============================] - 105s 87ms/step - loss: 0.1584 - accuracy: 0.9743 - val_loss: 0.6432 - val_accuracy: 0.9466
# Epoch 21/100
# 1215/1215 [==============================] - 106s 87ms/step - loss: 0.1420 - accuracy: 0.9754 - val_loss: 0.7562 - val_accuracy: 0.9614

# ########## When softmax, last layer =2, epochs=30, categorical
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.3971 - accuracy: 0.9514 - val_loss: 0.1750 - val_accuracy: 0.9622
# Epoch 2/30
# 1215/1215 [==============================] - 300s 247ms/step - loss: 0.1184 - accuracy: 0.9735 - val_loss: 0.2034 - val_accuracy: 0.9466
# Epoch 3/30
# 1215/1215 [==============================] - 301s 248ms/step - loss: 0.1012 - accuracy: 0.9770 - val_loss: 0.1238 - val_accuracy: 0.9658
# Epoch 4/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0999 - accuracy: 0.9751 - val_loss: 0.1992 - val_accuracy: 0.9308
# Epoch 5/30
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.0967 - accuracy: 0.9764 - val_loss: 0.1545 - val_accuracy: 0.9628
# Epoch 6/30
# 1215/1215 [==============================] - 300s 247ms/step - loss: 0.0899 - accuracy: 0.9771 - val_loss: 0.1902 - val_accuracy: 0.9518
# Epoch 7/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0821 - accuracy: 0.9787 - val_loss: 0.1566 - val_accuracy: 0.9654
# Epoch 8/30
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.0864 - accuracy: 0.9778 - val_loss: 0.2088 - val_accuracy: 0.9325
# Epoch 9/30
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.0807 - accuracy: 0.9786 - val_loss: 0.2961 - val_accuracy: 0.9320
# Epoch 10/30
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.0766 - accuracy: 0.9799 - val_loss: 0.1267 - val_accuracy: 0.9772
# Epoch 11/30
# 1215/1215 [==============================] - 301s 248ms/step - loss: 0.0771 - accuracy: 0.9790 - val_loss: 0.1371 - val_accuracy: 0.9727
# Epoch 12/30
# 1215/1215 [==============================] - 299s 246ms/step - loss: 0.0735 - accuracy: 0.9803 - val_loss: 0.1506 - val_accuracy: 0.9717
# Epoch 13/30
# 1215/1215 [==============================] - 304s 250ms/step - loss: 0.0709 - accuracy: 0.9807 - val_loss: 0.1781 - val_accuracy: 0.9705
# Epoch 14/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0750 - accuracy: 0.9807 - val_loss: 0.1824 - val_accuracy: 0.9593
# Epoch 15/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0728 - accuracy: 0.9811 - val_loss: 0.2135 - val_accuracy: 0.9626
# Epoch 16/30
# 1215/1215 [==============================] - 297s 245ms/step - loss: 0.0737 - accuracy: 0.9814 - val_loss: 0.1497 - val_accuracy: 0.9692
# Epoch 17/30
# 1215/1215 [==============================] - 297s 244ms/step - loss: 0.0675 - accuracy: 0.9826 - val_loss: 0.2744 - val_accuracy: 0.9096
# Epoch 18/30
# 1215/1215 [==============================] - 298s 246ms/step - loss: 0.0679 - accuracy: 0.9821 - val_loss: 0.2031 - val_accuracy: 0.9658
# Epoch 19/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0719 - accuracy: 0.9821 - val_loss: 0.2606 - val_accuracy: 0.9676
# Epoch 20/30
# 1215/1215 [==============================] - 298s 245ms/step - loss: 0.0697 - accuracy: 0.9822 - val_loss: 0.2069 - val_accuracy: 0.9669
# Epoch 21/30
# 1215/1215 [==============================] - 297s 245ms/step - loss: 0.0686 - accuracy: 0.9817 - val_loss: 0.2492 - val_accuracy: 0.9604
# Epoch 22/30
# 1215/1215 [==============================] - 297s 244ms/step - loss: 0.0704 - accuracy: 0.9827 - val_loss: 0.2292 - val_accuracy: 0.9432
# Epoch 23/30
# 1215/1215 [==============================] - 297s 244ms/step - loss: 0.0692 - accuracy: 0.9815 - val_loss: 0.2104 - val_accuracy: 0.9543
# Epoch 24/30
# 1215/1215 [==============================] - 296s 243ms/step - loss: 0.0653 - accuracy: 0.9824 - val_loss: 0.2229 - val_accuracy: 0.9403   
# Epoch 25/30
# 1215/1215 [==============================] - 295s 243ms/step - loss: 0.0682 - accuracy: 0.9828 - val_loss: 0.1587 - val_accuracy: 0.9682   
# Epoch 26/30
# 1215/1215 [==============================] - 296s 244ms/step - loss: 0.0651 - accuracy: 0.9823 - val_loss: 0.1822 - val_accuracy: 0.9660   
# Epoch 27/30
# 1215/1215 [==============================] - 296s 243ms/step - loss: 0.0650 - accuracy: 0.9826 - val_loss: 0.2972 - val_accuracy: 0.9079   
# Epoch 28/30
# 1215/1215 [==============================] - 296s 244ms/step - loss: 0.0622 - accuracy: 0.9824 - val_loss: 0.2011 - val_accuracy: 0.9758   
# Epoch 29/30
# 1215/1215 [==============================] - 296s 244ms/step - loss: 0.0640 - accuracy: 0.9831 - val_loss: 0.1747 - val_accuracy: 0.9718   
# Epoch 30/30
# 1215/1215 [==============================] - 297s 244ms/step - loss: 0.0651 - accuracy: 0.9820 - val_loss: 0.1543 - val_accuracy: 0.9728   
# Saved model to dis