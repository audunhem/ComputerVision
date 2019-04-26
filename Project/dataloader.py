from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import os
from random import shuffle
import numpy as np
import string
import random
import cv2

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
    return cv2.warpPerspective(image, M, (cols, rows)), vshift

def load_data():
    random.seed()
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'),  sep=",",names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
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
    camera_offset= 0.3
    for i in range(len(X_train)):
        if random.random() > 0.3 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][0])), cv2.COLOR_RGB2HSV))
            y_train_vals.append(y_train[i])
            if random.random() > 0.4:
                x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][1])), cv2.COLOR_RGB2HSV))
                y_train_vals.append(y_train[i]+camera_offset)
                x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][2])), cv2.COLOR_RGB2HSV))
                y_train_vals.append(y_train[i]-camera_offset)
            if(random.random() <= 0.2):
                    x_train_vals.append(transform_incline(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][1])), cv2.COLOR_RGB2HSV)))
                    y_train_vals.append(y_train[i])
            if(random.random() <= 0.0):
                    image, offset=transform_curvature(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][1])), cv2.COLOR_RGB2HSV))
                    x_train_vals.append(image)
                    y_train_vals.append(offset*0.4)
    for j in range(len(X_test)):
        x_test_vals.append(np.asarray(pyplot.imread(X_test[j][0])))
    for i in range(len(X_train)):
        if random.random() > 0.3 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(cv2.cvtColor(np.asarray(np.fliplr(pyplot.imread(X_train[i][0]))), cv2.COLOR_RGB2HSV))
            y_train_vals.append(-y_train[i])
            if random.random() > 0.4:
                x_train_vals.append(cv2.cvtColor(np.asarray(np.fliplr(pyplot.imread(X_train[i][1]))), cv2.COLOR_RGB2HSV))
                y_train_vals.append(-(y_train[i]+camera_offset))
                x_train_vals.append(cv2.cvtColor(np.asarray(np.fliplr(pyplot.imread(X_train[i][2]))), cv2.COLOR_RGB2HSV))
                y_train_vals.append(-(y_train[i]-camera_offset))

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
