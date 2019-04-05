import math
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation, Lambda, Reshape, Dropout
from dataloader import load_data
import numpy as np
from matplotlib import pyplot
import pandas as pd
import os
import string
import utils
import keract
from keract import get_activations
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def load_example():
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    for x in X:
        for i in range(3):
            folders = x[i].strip().split("/") #when running on unix-generated files
            #folders = x[i].split("\\") #when using windows-generated files
            x[i] = os.path.join(os.getcwd(),'recordings',folders[-2],folders[-1])
    example = pyplot.imread(X[2000][0])
    return example

if __name__ == '__main__':
    datagen = ImageDataGenerator(brightness_range=[0.02, 3])
    model = load_model('model.h5')

    example = load_example()
    example = np.resize(example,(1,160,320,3))
    example = datagen.flow(example[:,60:130,:,:])
    example = example[0]
    print(example.shape)

    a = keract.get_activations(model, example, layer_name='norm')

    #a = get_activations(model, example[:,60:130,:,:])
    keract.display_activations(a)
