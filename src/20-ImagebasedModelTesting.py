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
from keras.models import model_from_json

def load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, year, endYear):
    x_vali = []
    y_vali = []
    x_vali_benign = []
    y_vali_benign = []
    x_test = []
    y_test = []
    x_test_old = []
    y_test_old = []
    x_test_benign = []
    y_test_benign = []
    x_test_old_benign = []
    y_test_old_benign = []
    counter = 0
    for d in MalwareList:
        try:
            if d[2] >= year and  d[2] <= endYear:
                if counter == 4:
                    counter = 0
                    f = open(pathMalware+d[0],"rb")
                    img = pickle.load(f)
                    x_vali.append(img)
                    y_vali.append(1)
                counter += 1
            elif d[2] >= endYear:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            else:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(1)
        except:
            print("passed")
    counter = 0
    for d in range(len(BenignList)):
        if d >= (0.75*len(BenignList)):
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test_benign.append(img)
            y_test_benign.append(0)
        if d >= (0.50*len(BenignList)):
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test_old_benign.append(img)
            y_test_old_benign.append(0)
        else:
            if counter == 4:
                counter = 0
                f = open(pathBenign+BenignList[d],"rb")
                img = pickle.load(f)
                x_vali_benign.append(img)
                y_vali_benign.append(0)
            counter += 1

    x_vali = np.asarray(x_vali).astype('float32') / 255
    y_vali = np.asarray(y_vali)
    x_vali_benign = np.asarray(x_vali_benign).astype('float32') / 255
    y_vali_benign = np.asarray(y_vali_benign)
    x_test = np.asarray(x_test).astype('float32') / 255
    y_test = np.asarray(y_test)
    x_test_old = np.asarray(x_test_old).astype('float32') / 255
    y_test_old = np.asarray(y_test_old)
    x_test_benign = np.asarray(x_test_benign).astype('float32') / 255
    y_test_benign = np.asarray(y_test_benign)
    x_test_old_benign = np.asarray(x_test_old_benign).astype('float32') / 255
    y_test_old_benign = np.asarray(y_test_old_benign)
    return x_vali, y_vali, x_vali_benign, y_vali_benign,x_test,y_test,x_test_benign,y_test_benign,x_test_old,y_test_old,x_test_old_benign,y_test_old_benign


def load_data_Families(pathMalware, pathBenign, MalwareList, BenignList, year,TrainFamilies):
    x_test_in_family = []
    y_test_in_family = []
    x_test_out_family = []
    y_test_out_family = []

    x_test_sing = []
    y_test_sing = []


    counter = 0
    for d in MalwareList:
        try:
            if d[2] >= year and d[6] in TrainFamilies:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test_in_family.append(img)
                y_test_in_family.append(1)
            elif d[2] >= year and d[6] not in TrainFamilies and d[6] != "SINGLETON":
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test_out_family.append(img)
                y_test_out_family.append(1)
            elif d[2] >= year and d[6] == "SINGLETON":
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test_sing.append(img)
                y_test_sing.append(1)

        except:
            print("passed")


    x_test_in_family = np.asarray(x_test_in_family).astype('float32') / 255
    y_test_in_family = np.asarray(y_test_in_family)
    x_test_out_family = np.asarray(x_test_out_family).astype('float32') / 255
    y_test_out_family = np.asarray(y_test_out_family)
    x_test_sing = np.asarray(x_test_sing).astype('float32') / 255
    y_test_sing = np.asarray(y_test_sing)
    return x_test_in_family,y_test_in_family,x_test_out_family,y_test_out_family,x_test_sing,y_test_sing


def getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearstart,yearend):
    Families = []
    for d in MalwareList:
        try:
            if d[2] >= yearstart and  d[2] <= yearend and d[6] != "SINGLETON":
                Families.append(d[6])

        except:
            print("passed")
    Families = list(set(Families))
    return Families


pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_vali, y_vali, x_vali_benign, y_vali_benign, x_test, y_test,x_test_benign, y_test_benign, x_test_old, y_test_old,x_test_old_benign, y_test_old_benign =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)
print(x_test.shape)
print(y_test.shape)
print(x_test_old.shape)
print(y_test_old.shape)
print(x_test_benign.shape)
print(y_test_benign.shape)
print(x_test_old_benign.shape)
print(y_test_old_benign.shape)



x_vali = np.reshape(x_vali,(x_vali.shape[0],x_vali.shape[1],x_vali.shape[2],1))
x_vali_benign = np.reshape(x_vali_benign,(x_vali_benign.shape[0],x_vali_benign.shape[1],x_vali_benign.shape[2],1))
y_vali = keras.utils.to_categorical(y_vali, 2)
y_vali_benign = keras.utils.to_categorical(y_vali_benign, 2)


x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
x_test_old = np.reshape(x_test_old,(x_test_old.shape[0],x_test_old.shape[1],x_test_old.shape[2],1))
y_test = keras.utils.to_categorical(y_test, 2)
y_test_old = keras.utils.to_categorical(y_test_old, 2)

x_test_benign = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],x_test_benign.shape[2],1))
x_test_old_benign = np.reshape(x_test_old_benign,(x_test_old_benign.shape[0],x_test_old_benign.shape[1],x_test_old_benign.shape[2],1))
y_test_benign = keras.utils.to_categorical(y_test_benign, 2)
y_test_old_benign = keras.utils.to_categorical(y_test_old_benign, 2)




TrainFamilies = getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, 2017,2018)
x_test_in_family,y_test_in_family,x_test_out_family,y_test_out_family,x_test_sing,y_test_sing = load_data_Families(pathMalware, pathBenign, MalwareList, BenignList, 2019,TrainFamilies)
x_test_in_family = np.reshape(x_test_in_family,(x_test_in_family.shape[0],x_test_in_family.shape[1],x_test_in_family.shape[2],1))
y_test_in_family = keras.utils.to_categorical(y_test_in_family, 2)
x_test_out_family = np.reshape(x_test_out_family,(x_test_out_family.shape[0],x_test_out_family.shape[1],x_test_out_family.shape[2],1))
y_test_out_family = keras.utils.to_categorical(y_test_out_family, 2)
x_test_sing = np.reshape(x_test_sing,(x_test_sing.shape[0],x_test_sing.shape[1],x_test_sing.shape[2],1))
y_test_sing = keras.utils.to_categorical(y_test_sing, 2)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 20:
        lr *= 0.5e-1
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr





json_file = open('../Model/Image/Baseline.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Model/Image/Updated2017_.007.h5")



model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])



# Score trained model.
scores = model.evaluate(x_vali, y_vali, verbose=0)
print('Vali accuracy:', scores[1])
scores = model.evaluate(x_vali_benign, y_vali_benign, verbose=0)
print('Vali Benign accuracy:', scores[1])

scores = model.evaluate(x_test_old, y_test_old, verbose=0)
print('Test old accuracy:', scores[1])
scores = model.evaluate(x_test_old_benign, y_test_old_benign, verbose=0)
print('Test old Benign accuracy:', scores[1])

scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
scores = model.evaluate(x_test_benign, y_test_benign, verbose=0)
print('Test Benign accuracy:', scores[1])

scores = model.evaluate(x_test_in_family, y_test_in_family, verbose=0)
print('Test In Families accuracy:', scores[1])


scores = model.evaluate(x_test_out_family, y_test_out_family, verbose=0)
print('Test Out Families accuracy:', scores[1])

scores = model.evaluate(x_test_sing, y_test_sing, verbose=0)
print('Test Singleton accuracy:', scores[1])
