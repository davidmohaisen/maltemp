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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
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

def calculateConf(y_true,y_preds,y_preds_proba):
    conf = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            conf.append(y_preds_proba[i][1])
            # conf.append(max(0,2*(y_preds_proba[i][1]-0.5)))
    return 1.0*sum(conf)/len(conf)



json_file = open('../Model/Image/Baseline.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Model/Image/Updated2017_.007.h5")



model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])



yearStart = 2012
yearEnd = 2020

pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"

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

    x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2],1))
    y_test[i] = keras.utils.to_categorical(y_test[i], 2)
    y_true = np.argmax(y_test[i], axis = 1)
    y_preds = np.argmax(model.predict(x_test[i]), axis = 1)
    print(len(y_test[i]))
    print(len(y_true),len(y_preds))
    tpr = calculateEval(y_true,y_preds)
    y_preds_proba = model.predict(x_test[i])
    confedence = calculateConf(y_true,y_preds,y_preds_proba)
    confedenceMonthly.append(confedence)
    print("Month:",i,"TPR",tpr)
    tprMonthly.append(tpr)


path = "../Pickles/RetrainingMatrices/"

file = open(path+"ImageConfAVG","rb")
_,When = pickle.load(file)
print("=============================")

results = []
file = open(path+"Image","rb")
d = pickle.load(file)
currentI = 0
pointerI = 0
currentR = []
counter = 0

TimesRetrain = 0
for i in range(len(d)):
    currentR.append(d[currentI][i])
    if (pointerI < len(When) and i == When[pointerI]) or (counter != 0 and (counter%13) == 0):
        pointerI += 1
        currentI = i
        counter = 0
        TimesRetrain += 1
    else:
        counter += 1

ResultsMonthly = []
for i in np.arange(0,len(currentR),4):
    ResultsMonthly.append((currentR[i]+currentR[i+1]+currentR[i+2]+currentR[i+3])/4)
print(len(ResultsMonthly))

tprMonthlyRetrain = tprMonthly[:-21]
tprMonthlyRetrain = tprMonthlyRetrain + ResultsMonthly[:21]


x = np.linspace(0, len(tprMonthlyRetrain)-1, len(tprMonthlyRetrain))

z = np.polyfit(x, tprMonthlyRetrain, 3)
p = np.poly1d(z)
y = p(np.arange(0,len(tprMonthlyRetrain)))

margin = y-tprMonthlyRetrain
m = y-margin
# x = x[-21:]
# y = y[-21:]
# m = m[-21:]
plt.plot(x,y,"k-",linewidth=2,label='Retrain')
plt.fill_between(x, y , m, color="k", alpha=0.33)



x = np.linspace(0, len(tprMonthly)-1, len(tprMonthly))
z = np.polyfit(x, tprMonthly, 3)
p = np.poly1d(z)
y = p(np.arange(0,len(tprMonthly)))
margin = y-tprMonthly
for i in range(len(margin[:-21])):
    margin[i] = 0
m = y-margin
# x = x[-21:]
# y = y[-21:]

plt.plot(x,y,"r--",linewidth=2,label='Baseline')
plt.fill_between(x, y , m, color="r", alpha=0.33)



plt.xticks(labels = LabelString, ticks = indexLabel)
plt.xlabel("Time", fontsize=22)
plt.ylabel("Performance", fontsize=22)
plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
# plt.xlabel("Time", fontsize=18)
# plt.ylabel("Performance", fontsize=18)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
plt.ylim(0.00, 1.00)
plt.xlim(0, 102)
# plt.xlim(0, 126)
plt.legend(loc='lower right',  fontsize=18)
# plt.legend(loc='lower right',  fontsize=14)
plt.grid()
plt.show()
