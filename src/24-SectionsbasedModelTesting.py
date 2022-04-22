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
from sklearn.ensemble import RandomForestClassifier

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
                    f = open(pathMalware+d[0]+".pickle","rb")
                    img = pickle.load(f)
                    x_vali.append(img)
                    y_vali.append(1)
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
                x_test_benign.append(img)
                y_test_benign.append(0)
            if d >= (0.50*len(BenignList)):
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test_old_benign.append(img)
                y_test_old_benign.append(0)
            else:
                if counter == 4:
                    counter = 0
                    f = open(pathBenign+BenignList[d]+".pickle","rb")
                    img = pickle.load(f)
                    x_vali_benign.append(img)
                    y_vali_benign.append(0)
                counter += 1
        except:
            continue

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
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test_in_family.append(img)
                y_test_in_family.append(1)
            elif d[2] >= year and d[6] not in TrainFamilies and d[6] != "SINGLETON":
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test_out_family.append(img)
                y_test_out_family.append(1)
            elif d[2] >= year and d[6] == "SINGLETON":
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test_sing.append(img)
                y_test_sing.append(1)

        except:
            print("passed")

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

def vectorize(x,dic):
    new_x = []
    for i in range(len(x)):
        tmp = [0]*len(dic.keys())
        for key in x[i].keys():
            if key in dic:
                tmp[dic[key]] = 1
        new_x.append(tmp)
    return np.asarray(new_x)

pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/sections/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/sections/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_vali, y_vali, x_vali_benign, y_vali_benign, x_test, y_test,x_test_benign, y_test_benign, x_test_old, y_test_old,x_test_old_benign, y_test_old_benign =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)


f = open("../Pickles/SectionsDic","rb")
dic = pickle.load(f)

x_test = vectorize(x_test,dic)
x_test_old = vectorize(x_test_old,dic)
x_test_benign = vectorize(x_test_benign,dic)
x_test_old_benign = vectorize(x_test_old_benign,dic)
x_vali = vectorize(x_vali,dic)
x_vali_benign = vectorize(x_vali_benign,dic)








TrainFamilies = getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, 2017,2018)
x_test_in_family,y_test_in_family,x_test_out_family,y_test_out_family,x_test_sing,y_test_sing = load_data_Families(pathMalware, pathBenign, MalwareList, BenignList, 2019,TrainFamilies)

x_test_in_family = vectorize(x_test_in_family,dic)
x_test_out_family = vectorize(x_test_out_family,dic)
x_test_sing = vectorize(x_test_sing,dic)



f = open("../Model/Sections/RF","rb")
clf = pickle.load(f)


# Score trained model.
scores = clf.score(x_vali, y_vali)
print('Vali accuracy:', scores)
scores = clf.score(x_vali_benign, y_vali_benign)
print('Vali Benign accuracy:', scores)

scores = clf.score(x_test_old, y_test_old)
print('Test old accuracy:', scores)
scores = clf.score(x_test_old_benign, y_test_old_benign)
print('Test old Benign accuracy:', scores)

scores = clf.score(x_test, y_test)
print('Test accuracy:', scores)
scores = clf.score(x_test_benign, y_test_benign)
print('Test Benign accuracy:', scores)

scores = clf.score(x_test_in_family, y_test_in_family)
print('Test In Families accuracy:', scores)


scores = clf.score(x_test_out_family, y_test_out_family)
print('Test Out Families accuracy:', scores)

scores = clf.score(x_test_sing, y_test_sing)
print('Test Singleton accuracy:', scores)
