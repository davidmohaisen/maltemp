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
from sklearn.ensemble import RandomForestClassifier


def load_data_random(pathMalware, pathBenign, MalwareList, BenignList):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    malX = []
    malY = []
    for d in MalwareList:
        try:
            f = open(pathMalware+d[0]+".pickle","rb")
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
        try:
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


pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/symbols/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/symbols/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


x_train, y_train, x_test, y_test =  load_data_random(pathMalware, pathBenign, MalwareList, BenignList)

dic = getDic(x_train)

x_train = vectorize(x_train,dic)
x_test = vectorize(x_test,dic)




clf = RandomForestClassifier(random_state=0,class_weight={0: 1, 1: 1})
clf.fit(x_train, y_train)





# Score trained model.
scores = clf.score(x_train, y_train)
print('Train accuracy:', scores)
scores = clf.score(x_test, y_test)
print('Test accuracy:', scores)


f = open("../Model/Symbols/RF_Random","wb")
pickle.dump(clf,f)
