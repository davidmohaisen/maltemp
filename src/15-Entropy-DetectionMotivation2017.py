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

def load_data_random(pathMalware, pathBenign, MalwareList, BenignList):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    malX = []
    malY = []
    for d in MalwareList:
        try:
            f = open(pathMalware+d[0],"rb")
            img = pickle.load(f)
            malX.append(img)
            malY.append(1)
        except:
            print("passed")
    x_train, x_test, y_train, y_test = train_test_split(malX, malY, test_size=0.50, random_state=42)
    x_train = list(x_train)
    x_test = list(x_test)
    y_train = list(y_train)
    y_test = list(y_test)


    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test.append(img)
            y_test.append(0)
        else:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_train.append(img)
            y_train.append(0)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return x_train,y_train,x_test,y_test

def load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            index = int((d-(len(BenignList)/2))//step)
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test[index].append(img)
            y_test[index].append(0)
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


def load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            index = int((d-(len(BenignList)/2))//step)
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test[index].append(img)
            y_test[index].append(0)
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test

def PadZeroes(x_1,x_2,max=10000):
    n1 = []
    n2 = []
    # max = -1
    # for x in x_1:
    #     if len(x) > max:
    #         max = len(x)
    # for x in x_2:
    #     if len(x) > max:
    #         max = len(x)
    # if max > 10000:
    #     max = 10000
    for i in range(len(x_1)):

        if len(x_1[i]) >= max:
            n1.append(np.asarray(x_1[i][:max]))
        else :
            n1.append(np.asarray(list(x_1[i]) + [0]*(max-len(x_1[i]))))

    for i in range(len(x_2)):
        if len(x_2[i]) >= max:
            n2.append(np.asarray(x_2[i][:max]))
        else :
            n2.append(np.asarray(list(x_2[i]) + [0]*(max-len(x_2[i]))))
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)

    return n1, n2


pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/entropy/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/entropy/"
yearStart = 2019
yearEnd = 2020
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train, x_test, y_test =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart)

x_train,x_test = PadZeroes(x_train,x_test)



x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

model = Sequential()
model.add(Conv1D(32, kernel_size=(8), activation='relu', input_shape=x_train.shape[1:]))
# model.add(Dropout(0.25))
model.add(Conv1D(64, kernel_size=(8), activation='relu'))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_vali, y_vali))

scores = model.evaluate(x_train, y_train, verbose=1)
print('Train accuracy:', scores[1])
scores = model.evaluate(x_vali, y_vali, verbose=1)
print('Vali accuracy:', scores[1])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])



model_json = model.to_json()
with open("../Model/Entropy/Updated_2017.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save("../Model/Entropy/Updated_2017.h5")
print("Saved model to disk")
