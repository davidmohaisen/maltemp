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
import datetime
from sklearn.ensemble import RandomForestClassifier


def load_test_data_monthly_ExceptFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,FamilyAvoid,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])
    counter = 0
    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd  and d[6] not in FamilyAvoid and d[6] != "SINGLETON":
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


def load_test_data_weekly_ExceptFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,FamilyAvoid,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd and d[6] not in FamilyAvoid and d[6] != "SINGLETON":
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
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


def load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,FamilyInclude,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd:
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test



def load_test_data_monthly_OnlyFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,FamilyInclude,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd  and d[6] in FamilyInclude:
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


def load_test_data_weekly_OnlyFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,FamilyInclude,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd and d[6] in FamilyInclude:
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
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

def load_test_data_monthly_Singleton(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd  and d[6] == "SINGLETON":
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


def load_test_data_weekly_Singleton(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd,monitoredStart= 2017,monitoredEnd= 2018):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])
    counter = 0

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd  and d[6] == "SINGLETON":
                if  d[2] >= monitoredStart and d[2] <= monitoredEnd :
                    counter += 1
                    if counter != 4:
                        continue
                    elif counter ==4:
                        counter = 0
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

def getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearstart,yearend):
    Families = []
    for d in MalwareList:
        try:
            if d[2] >= yearstart and  d[2] <= yearend:
                Families.append(d[6])

        except:
            print("passed")
    Families = list(set(Families))
    return Families

def calculateEval(y_true,y_pred):
    p = 0
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            p += 1
            if y_true[i] == y_pred[i]:
                tp += 1

    tpr = 1.0*tp/p

    return tpr

def calculateConf(y_true,y_preds,y_preds_proba):
    conf = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            conf.append(max(0,2*(y_preds_proba[i][1]-0.5)))
    return 1.0*sum(conf)/len(conf)



yearStart = 2012
yearEnd = 2020


pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/hexdump/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/hexdump/"

f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)




f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

TrainFamilies = getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, 2017,2018)

x_test, y_test =  load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd,TrainFamilies)


f = open("../Model/Hexdump/RF","rb")
clf = pickle.load(f)


tprMonthly = []
confedenceMonthly = []
LabelString = []
indexLabel = []
for i in range(len(x_test)):
    if len(y_test[i]) < 10:
        continue
    if i%12 == 0:
        indexLabel.append(i)
        yearL = yearStart+(i//12)
        WeekL = (i%12)+1
        d = str(yearL)+'-'+str(WeekL)
        r = datetime.datetime.strptime(d, "%Y-%m")
        LabelString.append(r.strftime("%Y"))

    y_true = y_test[i]
    y_preds = clf.predict(x_test[i])
    print(len(y_test[i]))
    print(len(y_true),len(y_preds))
    tpr = calculateEval(y_true,y_preds)
    y_preds_proba = clf.predict_proba(x_test[i])
    confedence = calculateConf(y_true,y_preds,y_preds_proba)
    confedenceMonthly.append(confedence)
    print("Month:",i,"TPR",tpr)
    tprMonthly.append(tpr)

plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-k',label='TPR',linewidth=2);
plt.plot(np.arange(0,len(confedenceMonthly)), confedenceMonthly, '-r',label='Confidence',linewidth=2);
z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 3)
p = np.poly1d(z)
plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)
z = np.polyfit(np.arange(0,len(confedenceMonthly)), confedenceMonthly, 3)
p = np.poly1d(z)
plt.plot(np.arange(0,len(confedenceMonthly)),p(np.arange(0,len(confedenceMonthly))),"r--",linewidth=2)
plt.xticks(labels = LabelString, ticks = indexLabel)




plt.xlabel("Time", fontsize=18)
plt.ylabel("Performance", fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.00, 1.00)
plt.xlim(0, 102)
# plt.xlim(0, 126)
plt.legend(loc='lower right',  fontsize=12)

plt.show()
