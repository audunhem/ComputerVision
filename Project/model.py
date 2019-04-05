import math
from keras import models
from keras import layers
from keras import optimizers
from keract import get_activations
from keras.layers import Activation, Lambda, Reshape, Dropout
from dataloader import load_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



model_filename="dave2.json"
weights_filename="dave2.h5"

#Hyper parameters
epochs=10
batch_size=64

[x_train, x_test, y_train, y_test] = load_data()


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
x_train=x_train[:,60:130,:,:]


x_val = x_test[:,60:130,:,:];
y_val = y_test;


#x_val=x_train[-150:]
#y_val=y_train[-150:]
#print(x_train.shape)
#print(x_val.shape)


#x_train=x_train[:-150]
#y_train=y_train[:-150]
#print(x_train.shape)
x_test = np.array(x_test)
y_test = np.array(y_test)


datagen = ImageDataGenerator(brightness_range=[0.02, 3])

#Model
input_shape=(70,320,3)

model = models.Sequential()


#Normalization
model.add(layers.BatchNormalization(input_shape=input_shape, name='norm'))


#Convolutional layers
model.add(layers.Conv2D(filters=24,kernel_size=(5,5), strides=(2,2), name='conv1'))
model.add(Activation('relu', name='relu1'))
model.add(Dropout(0.5))

model.add(layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), name='conv2'))
model.add(Activation('relu', name='relu2'))
model.add(Dropout(0.5))

model.add(layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), name='conv3'))
model.add(Activation('relu', name='relu3'))
model.add(Dropout(0.5))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='conv4'))
model.add(Activation('relu', name='relu4'))
model.add(Dropout(0.1))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='conv5'))
model.add(Activation('relu', name='relu5'))
model.add(Dropout(0.5))

#Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(units=100, name='fc1'))
model.add(Activation('relu', name='relu6'))
model.add(Dropout(0.1))

model.add(layers.Dense(units=50, name='fc2'))
model.add(Activation('relu', name='relu7'))
model.add(Dropout(0.5))

model.add(layers.Dense(units=10, name='fc3'))
model.add(Activation('relu', name='relu8'))
model.add(Dropout(0.5))

model.add(layers.Dense(units=1, activation='tanh'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#model.fit(x_train, y_train, steps_per_epoch=300, epochs=epochs,validation_steps=40 ,validation_data=(x_val,y_val))
model.fit_generator(datagen.flow(x_train, y_train), samples_per_epoch=len(x_train), epochs=epochs, nb_val_samples = len(x_val), validation_data=(datagen.flow(x_val,y_val)))
#model.evaluate_generator(datagen.flow(x_test, y_test),steps=30)
#model.evaluate(x_test, y_test, batch_size=batch_size)
model.save('model.h5')
