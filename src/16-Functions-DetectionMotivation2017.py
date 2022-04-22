from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
from sklearn.utils import shuffle
import pickle
import requests
import json
import os
import subprocess
import cv2
import numpy
import pickle
import array
import os
import pickle
import os
import scipy
import array
import numpy as np
import scipy.misc
import imageio
from PIL import Image
from sklearn.model_selection import train_test_split
import collections
import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling2D
from sklearn.utils import class_weight
def load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, year):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for d in MalwareList:
        try:
            if d[2] >= year:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            elif d[2] <= 2017:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_train.append(img)
                y_train.append(1)
        except:
            print("passed")
    for d in range(len(BenignList)):
        try:
            BenignList[d] = BenignList[d].replace(" ","")
            if d >= len(BenignList)/2:
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(0)
            else:
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_train.append(img)
                y_train.append(0)
        except:
            continue
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return x_train,y_train,x_test,y_test

def getDic(x):
    dic = {}
    counter = 0
    for i in range(len(x)):
        for key in x[i].keys():
            if (not key in dic) and "fcn." not in key and "loc." not in key and "int." not in key:
                dic[key] = counter
                counter += 1

    return dic

def vectorize(x,dic):
    new_x = []
    for i in range(len(x)):
        tmp = [0]*len(dic.keys())
        for key in x[i].keys():
            if key in dic:
                tmp[dic[key]] = 1
        new_x.append(tmp)
    return np.asarray(new_x)

pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/functions/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/functions/"
yearStart = 2019
yearEnd = 2020
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train, x_test, y_test =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart)

dic = getDic(x_train)
print("finish dic initiation")
x_train = vectorize(x_train, dic)
x_test = vectorize(x_test, dic)

print(x_train.shape)
print(x_test.shape)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
#
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=x_train.shape[1:]))
# model.add(Dense(64, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(x_train, y_train,
          batch_size=32,
          epochs=20,
          validation_data=(x_vali, y_vali),
          class_weight=class_weights
          )

scores = model.evaluate(x_train, y_train, verbose=1)
print('Train accuracy:', scores[1])
scores = model.evaluate(x_vali, y_vali, verbose=1)
print('Vali accuracy:', scores[1])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])



model_json = model.to_json()
with open("../Model/Functions/Updated_2017.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save("../Model/Functions/Updated_2017.h5")
print("Saved model to disk")
