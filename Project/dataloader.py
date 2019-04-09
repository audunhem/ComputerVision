from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import os
from random import shuffle
import numpy as np
import string
import random


def load_data():
    random.seed()
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    for x in X:
        for i in range(3):


            folders = x[i].strip().split("/") #when running on unix-generated files
            #folders = x[i].split("\\") #when using windows-generated files
            x[i] = os.path.join(os.getcwd(),'recordings',folders[-2],folders[-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    x_train_vals = [];
    x_test_vals = [];
    y_train_vals = [];
    y_test = np.asarray(y_test)
    right_turns = 0
    left_turns = 0
    straight = 0
    for i in range(len(X_train)):
        if random.random() > 0.2 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(np.asarray(pyplot.imread(X_train[i][0])))
            y_train_vals.append(y_train[i])
            x_train_vals.append(np.asarray(pyplot.imread(X_train[i][1])))
            y_train_vals.append(y_train[i]+.25)
            x_train_vals.append(np.asarray(pyplot.imread(X_train[i][2])))
            y_train_vals.append(y_train[i]-.25)
    for j in range(len(X_test)):
        x_test_vals.append(np.asarray(pyplot.imread(X_test[j][0])))
    for i in range(len(X_train)):
        if random.random() > 0.2 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][0]))))
            y_train_vals.append(-y_train[i])
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][1]))))
            y_train_vals.append(-(y_train[i]+.25))
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][2]))))
            y_train_vals.append(-(y_train[i]-.25))

    """
    for i in range(len(X_train)):
        if random.random() > 0.2 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][0]))))
            y_train_vals.append(-y_train[i])
    """
    y_train_vals = np.asarray(y_train_vals)
    c = list(zip(x_train_vals, y_train_vals))
    random.shuffle(c)
    x_train_vals, y_train_vals = zip(*c)
    return x_train_vals, x_test_vals, y_train_vals, y_test

if __name__ == '__main__':
    [x_train, x_test, y_train, y_test] = load_data()
    np.savez('driving_data.npz', x_train=x_train, x_val=x_test, y_train=y_train, y_val=y_test) 
