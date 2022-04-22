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


def load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, year, endYear):
    x_train = []
    y_train = []
    x_vali = []
    y_vali = []
    x_test = []
    y_test = []
    x_test_old = []
    y_test_old = []
    counter = 0
    for d in MalwareList:
        try:
            if d[2] >= year and  d[2] <= endYear:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                if counter== 4:
                    counter = 0
                    x_vali.append(img)
                    y_vali.append(1)
                else :
                    x_train.append(img)
                    y_train.append(1)
                counter += 1
            elif d[2] >= endYear:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            else:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(1)
        except :
            print("passed")
    counter = 0
    for d in range(len(BenignList)):
        try:
            if d >= (0.75*len(BenignList)):
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(0)
            if d >= (0.50*len(BenignList)):
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(0)
            else:
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                if counter== 4:
                    counter = 0
                    x_vali.append(img)
                    y_vali.append(0)
                else :
                    x_train.append(img)
                    y_train.append(0)
                counter += 1
        except:
            print("passed")
    return x_train,y_train,x_vali, y_vali,x_test,y_test,x_test_old,y_test_old




def load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020):
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
            if d[2] >= yearStart and d[2] <= yearEnd:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


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
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train,x_vali, y_vali, x_test, y_test, x_test_old, y_test_old =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)


f = open("../Pickles/FunctionsDic","rb")
dic = pickle.load(f)

x_train = vectorize(x_train,dic)



x_test, y_test =  load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020)
x_test = x_test[:92]
y_test = y_test[:92]



matrixToCal = []
for i in range(len(x_test)):
    matrixToCal.append([])

    f = open("../Model/Functions/Model_Week_"+str(i),"rb")
    clf = pickle.load(f)

    for j in range(len(x_test)):
        score = 0
        if len(x_test[j]) == 0:
            if j == 0:
                score = 1.0
            else:
                score = matrixToCal[i][j-1]
        else:
            x_test_n = vectorize(x_test[j],dic)
            y_test_n = y_test[j]
            score = clf.score(x_test_n, y_test_n)
        matrixToCal[-1].append(score)
        print(i,j,score)

f = open("../Pickles/RetrainingMatrices/Functions","wb")
print(matrixToCal)
pickle.dump(matrixToCal,f)
