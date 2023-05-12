
import  tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from keras.layers import Input,Lambda ,Dense ,Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
from PIL import Image

train_path = 'P:\\Machine Learning\\Plant diseases\\Data set\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train'
valid_path = 'P:\\Machine Learning\\Plant diseases\\Data set\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\valid'
IMAGE_SIZE=[224,224]

inception =InceptionV3(input_shape=IMAGE_SIZE +[3],weights='imagenet',include_top=False)

for layers in inception.layers:
  layers.trainable=False

folders= glob(train_path+'/*')
# print(len(folders))

x=Flatten()(inception.output)
prediction =Dense(len(folders),activation='softmax')(x)
model =Model(inputs=inception.input,outputs=prediction)
# print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range =0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set =train_datagen.flow_from_directory(train_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical'
                                                )
# print(training_set)

test_set =train_datagen.flow_from_directory(valid_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical'
                                                )
# print(test_set)

r=model.fit  (
    training_set,validation_data=test_set,epochs=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)

)
# model.save('abc.h5')