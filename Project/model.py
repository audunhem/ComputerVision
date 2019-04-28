import math
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation, Lambda, Reshape, Dropout
from dataloader import load_data
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
import cv2


#Hyper parameters
epochs=20
batch_size=64

#Loading of data
data = np.load('driving_data.npz')
x_train = np.array(data['x_train'])
x_val = np.array(data['x_val'])
y_train = np.array(data['y_train'])
y_val = np.array(data['y_val'])

#Cropping
x_train=x_train[:,70:140,40:280,:]
x_val = x_val[:,70:140,40:280,:]


num_samples = y_train.size
steps = np.ceil(num_samples / batch_size)


#Preprocces function that is appliet to each input to the network
def pre_process(image):
        #Add random brightness to image
        if(random.random() <= 0.4):
                image =np.array(image)
                bright_factor = random.uniform(0.7,1.3)
                #image[:,:,2] = image[:,:,2]*bright_factor
        #Add a random shadow to image
        if(random.random() <= 1):
                bright_factor = random.uniform(0.4,0.8)
                x = random.randint(0, image.shape[1])
                y = random.randint(0, image.shape[0])
                width = 0
                height = 0
                rand_var = random.randint(1,2)
                if rand_var == 1:
                    x = 0
                    width = image.shape[1]
                    if y < image.shape[0]/2:
                        height = y
                        y = 0
                    else:
                        height = image.shape[0]-y

                if rand_var == 2:
                    y = 0
                    height = image.shape[0]
                    if x < image.shape[1]/2:
                        width = x
                        x = 0
                    else:
                        width = image.shape[1]-x
                #Assuming HSV image
                image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor

        return image

datagen = ImageDataGenerator(preprocessing_function=pre_process)




#Model
input_shape=(70,240,3)

model = models.Sequential()


#Normalization
model.add(layers.BatchNormalization(input_shape=input_shape, name='norm'))


#Convolutional layers
model.add(layers.Conv2D(filters=24,kernel_size=(5,5), strides=(2,2), name='conv1'))
model.add(Activation('relu', name='relu1'))


model.add(layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), name='conv2'))
model.add(Activation('relu', name='relu2'))


model.add(layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), name='conv3'))
model.add(Activation('relu', name='relu3'))


model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='conv4'))
model.add(Activation('relu', name='relu4'))


model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='conv5'))
model.add(Activation('relu', name='relu5'))


#Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(units=100, name='fc1'))
model.add(Activation('relu', name='relu6'))


model.add(layers.Dense(units=50, name='fc2'))
model.add(Activation('relu', name='relu7'))

model.add(layers.Dense(units=10, name='fc3'))
model.add(Activation('relu', name='relu8'))


model.add(layers.Dense(units=1, activation='tanh'))


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=steps, epochs=epochs, validation_steps = 50, validation_data=(datagen.flow(x_val,y_val)))
model.save('model.h5')
