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
    x_vali = []
    y_vali = []

    counter = 0
    for d in range(len(BenignList)):
        try:
            if d >= (0.75*len(BenignList)):
                continue
            if d >= (0.50*len(BenignList)):
                continue
            else:
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                if counter!= 4:
                    x_vali.append(img)
                    y_vali.append(0)
                    counter = 0
                counter += 1
        except:
            print("passed")
    return x_vali, y_vali





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

x_vali, y_vali =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)


f = open("../Pickles/SectionsDic","rb")
dic = pickle.load(f)

x_vali = vectorize(x_vali,dic)





matrixToCal = []
for i in range(92):

    f = open("../Model/Sections/Model_Week_"+str(i),"rb")
    clf = pickle.load(f)

    x_test_n = x_vali
    y_test_n = y_vali
    score = clf.score(x_test_n, y_test_n)
    matrixToCal.append(score)
    print(i,score)

f = open("../Pickles/RetrainingMatrices/SectionsBenign","wb")
print(matrixToCal)
pickle.dump(matrixToCal,f)

print(round(100*matrixToCal[0],2),"&",round(100*matrixToCal[13],2),"&",round(100*matrixToCal[26],2),"&",round(100*matrixToCal[52],2),"&",round(100*matrixToCal[-1],2))
