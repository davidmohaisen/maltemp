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
        except:
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
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_vali = np.asarray(x_vali)
    y_vali = np.asarray(y_vali)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_test_old = np.asarray(x_test_old)
    y_test_old = np.asarray(y_test_old)
    return x_train,y_train,x_vali, y_vali,x_test,y_test,x_test_old,y_test_old

def load_test_data_weekly(TakenFamilies, pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020):
    x_test_in = []
    y_test_in = []
    x_test_out = []
    y_test_out = []
    x_test_sing = []
    y_test_sing = []

    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test_in.append([])
            y_test_in.append([])
            x_test_out.append([])
            y_test_out.append([])
            x_test_sing.append([])
            y_test_sing.append([])


    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd:
                if d[6] in TrainFamilies:
                    f = open(pathMalware+d[0]+".pickle","rb")
                    img = pickle.load(f)
                    index = ((d[2]-yearStart)*53)+d[4]-1
                    x_test_in[index].append(img)
                    y_test_in[index].append(1)
                elif d[6] not in TrainFamilies and d[6] != "SINGLETON":
                    f = open(pathMalware+d[0]+".pickle","rb")
                    img = pickle.load(f)
                    index = ((d[2]-yearStart)*53)+d[4]-1
                    x_test_out[index].append(img)
                    y_test_out[index].append(1)
                elif d[6] == "SINGLETON":
                    f = open(pathMalware+d[0]+".pickle","rb")
                    img = pickle.load(f)
                    index = ((d[2]-yearStart)*53)+d[4]-1
                    x_test_sing[index].append(img)
                    y_test_sing[index].append(1)

        except:
            print("passed")


    for i in range(len(x_test_in)):
        x_test_in[i] = np.asarray(x_test_in[i])
        y_test_in[i] = np.asarray(y_test_in[i])
    for i in range(len(x_test_out)):
        x_test_out[i] = np.asarray(x_test_out[i])
        y_test_out[i] = np.asarray(y_test_out[i])
    for i in range(len(x_test_sing)):
        x_test_sing[i] = np.asarray(x_test_sing[i])
        y_test_sing[i] = np.asarray(y_test_sing[i])
    return x_test_in,y_test_in,x_test_out,y_test_out,x_test_sing,y_test_sing




pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/hexdump/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/hexdump/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train,x_vali, y_vali, x_test, y_test, x_test_old, y_test_old =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)




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



TrainFamilies = getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, 2017,2018)


x_test_in,y_test_in,x_test_out,y_test_out,x_test_sing,y_test_sing =  load_test_data_weekly(TrainFamilies,pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020)
x_test_in = x_test_in[:92]
y_test_in = y_test_in[:92]
x_test_out = x_test_out[:92]
y_test_out = y_test_out[:92]
x_test_sing = x_test_sing[:92]
y_test_sing = y_test_sing[:92]

matrixToCal_in = []
matrixToCal_out = []
matrixToCal_sing = []

for i in range(len(x_test_in)):
    matrixToCal_in.append([])
    matrixToCal_out.append([])
    matrixToCal_sing.append([])

    f = open("../Model/Hexdump/Model_Week_"+str(i),"rb")
    clf = pickle.load(f)

    for j in range(len(x_test_in)):
        score_in = 0
        if len(x_test_in[j]) == 0:
            if j == 0:
                score_in = 1.0
            else:
                score_in = matrixToCal_in[i][j-1]
        else:
            x_test_n = x_test_in[j]
            y_test_n = y_test_in[j]
            score_in = clf.score(x_test_n, y_test_n)
        matrixToCal_in[-1].append(score_in)


        score_out = 0
        if len(x_test_out[j]) == 0:
            if j == 0:
                score_out = 1.0
            else:
                score_out = matrixToCal_out[i][j-1]
        else:
            x_test_n = x_test_out[j]
            y_test_n = y_test_out[j]
            score_out = clf.score(x_test_n, y_test_n)
        matrixToCal_out[-1].append(score_out)

        score_sing = 0
        if len(x_test_sing[j]) == 0:
            if j == 0:
                score_sing = 1.0
            else:
                score_sing = matrixToCal_sing[i][j-1]
        else:
            x_test_n = x_test_sing[j]
            y_test_n = y_test_sing[j]
            score_sing = clf.score(x_test_n, y_test_n)
        matrixToCal_sing[-1].append(score_sing)



f = open("../Pickles/RetrainingMatrices/HexdumpSplitData","wb")
print(matrixToCal_in)
print(matrixToCal_out)
print(matrixToCal_sing)
pickle.dump([matrixToCal_in,matrixToCal_out,matrixToCal_sing],f)
