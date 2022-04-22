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
            else:
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




pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/hexdump/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/hexdump/"
yearStart = 2019
yearEnd = 2020
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train, x_test, y_test =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart)


#
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.20, random_state=10)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

scores = clf.score(x_train, y_train)
print('Train accuracy:', scores)
scores = clf.score(x_vali, y_vali)
print('Vali accuracy:', scores)
scores = clf.score(x_test, y_test)
print('Test accuracy:', scores)



f = open("../Model/Hexdump/BaselineRF.pickle","wb")
pickle.dump(clf,f)
