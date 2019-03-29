from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import os
from random import shuffle


def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ind_list = [i for i in range(len(X_train))]
    shuffle(ind_list)
    train_new  = X_train[ind_list]
    target_new = y_train[ind_list]
    x_train_vals = [];
    x_test_vals = [];
    for i in range(len(X_train)):
        x_train_vals.append(pyplot.imread(X_train[i][0]))
    for j in range(len(X_test)):
        x_test_vals.append(pyplot.imread(X_test[j][0]))
    return x_train_vals, x_test_vals, y_train, y_test

#image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
