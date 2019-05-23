'''
Filename: e:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360/v1_fruit_classify_with_pretrainedmodel.py
Path: e:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360
Created Date: Tuesday, May 21st 2019, 7:39:33 pm
Author: apotdar
'''

import os

from tensorflow.keras import models  
from tensorflow.keras import layers  
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getAllClassNames(dir_path):
    return os.listdir(dir_path)

base_dir_path = 'E:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360/'
train_dir_path = os.path.join(base_dir_path,'train')
test_dir_path = os.path.join(base_dir_path,'test')

AllClassNames = getAllClassNames(train_dir_path)
DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))  
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
           
model.summary()

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_generator = datagen.flow_from_directory(
        train_dir_path,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

test_generator = datagen.flow_from_directory(
        test_dir_path,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

model.fit_generator(train_generator,
                    epochs=50,
                    validation_data = test_generator)
