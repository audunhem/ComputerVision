import math
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation, Lambda, Reshape, Dropout
from dataloader import load_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import string
import utils
import keract
import cv2
import random
from keract import get_activations
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def pre_process(image):

        if(random.random() <= 0.4):
                image =np.array(image)
                bright_factor = random.uniform(0.1,0.4)
                image[:,:,2] = image[:,:,2]*bright_factor
        if(random.random() <= 0.0):
                image =np.array(image)
                sat_factor = random.uniform(0.2,1.6)
                image[:,:,1] = image[:,:,1]*sat_factor
        if(random.random() <= 0):
                image=transform_curvature(image)
        if(random.random() <= 0.8):
                bright_factor = random.uniform(0.2,0.8)
                #print(image.shape)
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
                    #print(y, height)

                if rand_var == 2:
                    y = 0
                    height = image.shape[0]
                    if x < image.shape[1]/2:
                        width = x
                        x = 0
                    else:
                        width = image.shape[1]-x
                    #print(x, width)
                #Assuming HSV image
                image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor
        return image


def transform_incline(image, shift=(0.1,0.6), orientation='rand'):

    rows,cols,ch = image.shape

    vshift = random.uniform(shift[0],shift[1])
    if orientation == 'rand':
        orientation = random.choice(['down', 'up'])
    if orientation == 'up':
        vshift = -vshift*0.2
    elif orientation != 'down':
        raise ValueError("No or unknown orientation given. Possible values are 'up' and 'down'.")

    dst = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) * np.float32([cols, rows])

    src = np.float32([[0., 0.], [1., 0.], [0-vshift, 1], [1+vshift, 1]]) * np.float32([cols, rows])
    #Calculate the transformation matrix, perform the transformation,
    #and return it.
    M = cv2.getPerspectiveTransform(src, dst)
    print(cv2.warpPerspective(image, M, (cols, rows)).shape)
    return cv2.warpPerspective(image, M, (cols, rows))


def transform_curvature(image, shift=(0.1,0.8), orientation='rand'):

    rows,cols,ch = image.shape

    vshift = random.uniform(shift[0],shift[1])

    if orientation == 'rand':
        orientation = random.choice(['left', 'right'])

    if orientation == 'left':
        vshift = -vshift
    elif orientation != 'right':
        raise ValueError("No or unknown orientation given. Possible values are 'left' and 'right'.")

    dst = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) * np.float32([cols, rows])

    src = np.float32([[0., 0.], [1., 0.], [0+vshift, 1], [1+vshift, 1]]) * np.float32([cols, rows])

    #Calculate the transformation matrix, perform the transformation,
    #and return it.
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (cols, rows))


def load_example():
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    for x in X:
        for i in range(3):
            folders = x[i].strip().split("/") #when running on unix-generated files
            #folders = x[i].split("\\") #when using windows-generated files
            x[i] = os.path.join(os.getcwd(),'recordings',folders[-2],folders[-1])
    example = cv2.cvtColor(np.asarray(plt.imread(X[300][0])), cv2.COLOR_BGR2HSV)

    return example

def plot_histogram():
    data = np.load('driving_data.npz')
    y_train = np.array(data['y_train'])
    y_val = np.array(data['y_val'])
    plt.hist(y_train, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

if __name__ == '__main__':
    random.seed()
    datagen = ImageDataGenerator(featurewise_center=True,brightness_range=[0.02, 2],height_shift_range=0, preprocessing_function = pre_process)
    model = load_model('model.h5')

    example = load_example()
    example = np.resize(example,(1,160,320,3))
    example = datagen.flow(x=example[:,70:140,:,:])
    #example = datagen.flow(example[:,:,:,:])
    example = example[0]
    #example = example[:,70:140,40:280,:]
    print(example)


    a = keract.get_activations(model, example, layer_name='norm')

    #a = get_activations(model, example[:,60:130,:,:])
    keract.display_activations(a)
    plot_histogram()
