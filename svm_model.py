import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from keras.regularizers import l2
from glob import glob

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

folders= glob(train_path+'/*')

model = keras.models.Sequential()
model.add(Conv2D(filters=32, padding='same', kernel_size=12, activation='relu', strides=4, input_shape=[224, 224]+[3]))
model.add(MaxPool2D(pool_size=9, strides=3))
model.add(Flatten())
model.add(Dense(units=448, activation='relu'))
model.add(Dense(len(folders), kernel_regularizer=l2(0.01), activation='softmax'))

model.compile(optimizer='adam',loss='squared_hinge',metrics=['accuracy'])

model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)

)

model.save('plant_disease_svm.h5')