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
import sklearn
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling2D

def calculateEval(y_true,y_pred):
    p = 0
    tp = 0
    n = 0
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            n += 1
            if y_true[i] == y_pred[i]:
                tn += 1
        else:
            p += 1
            if y_true[i] == y_pred[i]:
                tp += 1
    tnr = 1.0*tn/n
    tpr = 1.0*tp/p
    # fscore = tp/(tp+0.5*((p-tp)+(n-tn)))
    fscore = (sklearn.metrics.f1_score(y_true,y_pred)+sklearn.metrics.f1_score(y_true,y_pred,pos_label=0))/2
    acc = (tp+tn)/(p+n)
    return acc,tnr,tpr,fscore

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
            elif d[2] >= 2017:
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

# def load_data_random(pathMalware, pathBenign, MalwareList, BenignList):
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []
#     malX = []
#     malY = []
#     for d in MalwareList:
#         try:
#             f = open(pathMalware+d[0],"rb")
#             img = pickle.load(f)
#             malX.append(img)
#             malY.append(1)
#         except:
#             print("passed")
#     x_train, x_test, y_train, y_test = train_test_split(malX, malY, test_size=0.50, random_state=42)
#     x_train = list(x_train)
#     x_test = list(x_test)
#     y_train = list(y_train)
#     y_test = list(y_test)
#
#
#     for d in range(len(BenignList)):
#         if d >= len(BenignList)/2:
#             f = open(pathBenign+BenignList[d],"rb")
#             img = pickle.load(f)
#             x_test.append(img)
#             y_test.append(0)
#         else:
#             f = open(pathBenign+BenignList[d],"rb")
#             img = pickle.load(f)
#             x_train.append(img)
#             y_train.append(0)
#     x_train = np.asarray(x_train).astype('float32') / 255
#     y_train = np.asarray(y_train)
#     x_test = np.asarray(x_test).astype('float32') / 255
#     y_test = np.asarray(y_test)
#     return x_train,y_train,x_test,y_test

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
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        try:
            BenignList[d] = BenignList[d].replace(" ","")
            if d >= len(BenignList)/2:
                index = int((d-(len(BenignList)/2))//step)
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test[index].append(img)
                y_test[index].append(0)
        except:
            continue
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
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
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        try:
            BenignList[d] = BenignList[d].replace(" ","")
            if d >= len(BenignList)/2:
                index = int((d-(len(BenignList)/2))//step)
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test[index].append(img)
                y_test[index].append(0)
        except:
            continue
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test

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


x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

f = open("../Model/Functions/BaselineRF.pickle","rb")
model = pickle.load(f)




x_test, y_test =  load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
tprMonthly = []
LabelString = []
indexLabel = []
for i in range(len(x_test)):
    if 1 not in collections.Counter(y_test[i]).keys():
        continue
    x_test[i] = vectorize(x_test[i], dic)

    if i%3 == 0:
        indexLabel.append(i)
        yearL = 2019+(i//12)
        WeekL = (i%12)+1
        d = str(yearL)+'-'+str(WeekL)
        r = datetime.datetime.strptime(d, "%Y-%m")
        LabelString.append(r.strftime("%b %y"))

    x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1]))
    y_true = y_test[i]
    y_preds = model.predict(x_test[i])
    acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
    print("Month:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
    tprMonthly.append(tpr)

plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-ok');
z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 1)
p = np.poly1d(z)
plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)
plt.xticks(labels = LabelString, ticks = indexLabel)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Performance", fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.40, 1.00)

plt.show()

x_test, y_test =  load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
tprWeekly = []
LabelString = []
indexLabel = []
for i in range(len(x_test)):
    if 1 not in collections.Counter(y_test[i]).keys():
        continue
    x_test[i] = vectorize(x_test[i], dic)
    if i%15 == 0:
        indexLabel.append(i)
        yearL = 2019+(i//53)
        WeekL = (i%53)+1
        d = str(yearL)+'-W'+str(WeekL)
        # r = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
        r = datetime.datetime.strptime(d+"-1", "%Y-W%W-%w")

        LabelString.append(r.strftime("%b %y"))
    x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1]))
    y_true = y_test[i]
    y_preds = model.predict(x_test[i])
    acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
    print("Week:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
    tprWeekly.append(tpr)

plt.plot(np.arange(0,len(tprWeekly)), tprWeekly, '-ok');
z = np.polyfit(np.arange(0,len(tprWeekly)), tprWeekly, 1)
p = np.poly1d(z)
plt.plot(np.arange(0,len(tprWeekly)),p(np.arange(0,len(tprWeekly))),"k--",linewidth=2)

plt.xticks(labels = LabelString, ticks = indexLabel)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Performance", fontsize=18)

plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.40, 1.00)

plt.show()
