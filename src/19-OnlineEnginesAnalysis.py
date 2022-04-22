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


def load_test_data_monthly(MalwareList,Detection, yearStart,yearEnd):
    DetectedRatio = []
    DetectionPerEngineRatio = []

    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            DetectedRatio.append([])
            DetectionPerEngineRatio.append([])
    for d in range(len(MalwareList)):
        if MalwareList[d][2] >= yearStart and MalwareList[d][2] <= yearEnd:
            index = ((MalwareList[d][2]-yearStart)*12)+MalwareList[d][3]-1
            if Detection[d][1] == 0:
                DetectedRatio[index].append(0)
                DetectionPerEngineRatio[index].append(0)
            else:
                DetectedRatio[index].append(1)
                DetectionPerEngineRatio[index].append(Detection[d][2])

    return DetectedRatio, DetectionPerEngineRatio





yearStart = 2012
yearEnd = 2020



f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


f = open("../Pickles/PaddedReports","rb")
Detection = pickle.load(f)




DetectedRatio, DetectionPerEngineRatio =  load_test_data_monthly(MalwareList,Detection, yearStart, yearEnd)



tprMonthly = []
tprMonthlyPerEngine = []
LabelString = []
indexLabel = []
for i in range(len(DetectedRatio)):
    if len(DetectedRatio[i]) < 10:
        continue
    if i%12 == 0:
        indexLabel.append(i)
        yearL = yearStart+(i//12)
        WeekL = (i%12)+1
        d = str(yearL)+'-'+str(WeekL)
        r = datetime.datetime.strptime(d, "%Y-%m")
        LabelString.append(r.strftime("%Y"))


    tpr = 1.0*sum(DetectedRatio[i])/len(DetectedRatio[i])
    print("Month:",i,"TPR",tpr)
    tprMonthly.append(tpr)
    tpr = 1.0*sum(DetectionPerEngineRatio[i])/len(DetectionPerEngineRatio[i])
    print("Month:",i,"TPR",tpr)
    tprMonthlyPerEngine.append(tpr)

# plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-ok');
# z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 3)
# p = np.poly1d(z)
# plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)


plt.plot(np.arange(0,len(tprMonthlyPerEngine)), tprMonthlyPerEngine, '-ok');
z = np.polyfit(np.arange(0,len(tprMonthlyPerEngine)), tprMonthlyPerEngine, 5)
p = np.poly1d(z)
plt.plot(np.arange(0,len(tprMonthlyPerEngine)),p(np.arange(0,len(tprMonthlyPerEngine))),"k--",linewidth=2)


plt.xticks(labels = LabelString, ticks = indexLabel)


plt.xlabel("Time", fontsize=18)
plt.ylabel("Average Detection Rate", fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.00, 1.00)
plt.xlim(0, 102)
# plt.xlim(0, 78)
# plt.legend(loc='upper right',  fontsize=12)

plt.show()
