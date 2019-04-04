from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import os
from random import shuffle
import numpy as np
import string


def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    for x in X:
        for i in range(3):
            folders = x[i].split("/") #when running on unix-generated files
            #folders = x[i].split("\\") #when using windows-generated files
            x[i] = os.path.join(os.getcwd(),'recordings',folders[-2],folders[-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ind_list = [i for i in range(len(X_train))]
    shuffle(ind_list)
    X_train  = X_train[ind_list]
    y_train = y_train[ind_list]
    x_train_vals = [];
    x_test_vals = [];
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    right_turns = 0
    left_turns = 0
    straight = 0
    for i in range(len(X_train)):
        if right_turns > left_turns and y_train[i] > 0:
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][0]))))
            y_train[i] = -y_train[i]
            left_turns = left_turns + 1
        elif right_turns < left_turns and y_train[i] < 0:
            x_train_vals.append(np.asarray(np.fliplr(pyplot.imread(X_train[i][0]))))
            y_train[i] = -y_train[i]
            right_turns = right_turns + 1
        else:
            x_train_vals.append(np.asarray(pyplot.imread(X_train[i][0])))
            if y_train[i] < 0:
                left_turns = left_turns + 1
            elif y_train[i] > 0:
                right_turns = right_turns + 1

    for j in range(len(X_test)):
        x_test_vals.append(np.asarray(pyplot.imread(X_test[j][0])))
    return x_train_vals, x_test_vals, y_train, y_test
