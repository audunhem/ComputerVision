import math
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation

#Hyper parameters
epochs=10
batch_size=64


#Model
input_shape=(150,150,3)

model = models.Sequential()

#Normalization
model.add(layers.BatchNormalization(input_shape=input_shape))

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
model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, validation_data=(x_val,y_val))
model.evaluate(x_test, y_test, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch = ceil(num_train_samples / batch_size), epochs=epochs, valitation_data=val_generator,validation_steps= ceil(val_samples / batch_size))
