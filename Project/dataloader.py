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

def load_data():
    #reading csv file with image data
    data_df = pd.read_csv(os.path.join(os.getcwd(),'recordings', 'driving_log.csv'),  sep=",",names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values #array of file paths
    y = data_df['steering'].values #array of

    #stripping file paths to get only the last folders
    for x in X:
        for i in range(3):
            folders = x[i].strip().split("/") #when running on unix-generated files
            #folders = x[i].split("\\") #when using windows-generated files
            x[i] = os.path.join(os.getcwd(),'recordings',folders[-2],folders[-1])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    #arrays to store the numerical values
    x_train_vals = [];
    x_val_vals = [];
    y_train_vals = [];
    y_val = np.asarray(y_val)
    camera_offset= 0.3 #steering angle offset for side images
    for i in range(len(X_train)):
        #skipping 30% of the straight images
        if random.random() > 0.3 and y_train[i] == 0:
            #do nothing
            continue
        else:
            x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][0])), cv2.COLOR_RGB2HSV))
            y_train_vals.append(y_train[i])
            #also using 60% of the side images
            if random.random() > 0.4:
                x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][1])), cv2.COLOR_RGB2HSV))
                y_train_vals.append(y_train[i]+camera_offset)
                x_train_vals.append(cv2.cvtColor(np.asarray(pyplot.imread(X_train[i][2])), cv2.COLOR_RGB2HSV))
                y_train_vals.append(y_train[i]-camera_offset)
    #adding values to the test set
    for j in range(len(X_val)):
        x_val_vals.append(np.asarray(cv2.cvtColor(np.asarray(pyplot.imread(X_val[j][0]), cv2.COLOR_RGB2HSV))))

    #doing the same as above, only with flipping the images
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

    #converting to numpy array and shuffling training values
    y_train_vals = np.asarray(y_train_vals)
    shuffle_set = list(zip(x_train_vals, y_train_vals))
    random.shuffle(shuffle_set)
    x_train_vals, y_train_vals = zip(*shuffle_set)
    return x_train_vals, x_val_vals, y_train_vals, y_val


if __name__ == '__main__':
    random.seed()
    #loading the recording data
    [x_train, x_test, y_train, y_val] = load_data()
    #saving the data locally
    np.savez('driving_data.npz', x_train=x_train, x_val=x_test, y_train=y_train, y_val=y_val)
