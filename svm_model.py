import keras.models
import tensorflow as tf
from keras.preprocessing.image import  ImageDataGenerator
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from keras.regularizers import l2

train_path = 'P:\\Machine Learning\\Plant diseases\\Data set\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train'
valid_path = 'P:\\Machine Learning\\Plant diseases\\Data set\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\valid'

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set =train_datagen.flow_from_directory(train_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical'
                                                )
test_set =train_datagen.flow_from_directory(valid_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical'
                                                )
# print(test_set)
# print(training_set)

cnn = keras.models.Sequential()
cnn.add(Conv2D(filters=32,padding='same',kernel_size=12,activation='relu',strides=4,input_shape=[224,224]))
cnn.add(MaxPool2D(pool_size=9,strides=3))
cnn.add(Flatten())
cnn.add(Dense(units=448,activation='relu'))
