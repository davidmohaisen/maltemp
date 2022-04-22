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

def load_data_time_at_once(pathMalware, MalwareList, year, endYear):
    x=[]
    details = []
    for d in MalwareList:
        try:
            if d[2] >= year and  d[2] <= endYear:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x.append(img)
                details.append(d)

        except :
            pass

    return x,details


def vectorize(x,dic):
    new_x = []
    for i in range(len(x)):
        tmp = [0]*len(dic.keys())
        for key in x[i].keys():
            if key in dic:
                tmp[dic[key]] = 1
        new_x.append(tmp)
    return np.asarray(new_x)




pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/relocs/"
yearStart = 2008
yearEnd = 2020

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x,details =  load_data_time_at_once(pathMalware, MalwareList, yearStart,yearEnd)


f = open("../Pickles/RelocsDic","rb")
dic = pickle.load(f)

x = vectorize(x,dic)



print(len(x))



uniques = np.unique(x, axis=0)
print(len(uniques))
print(uniques[0])

indeces = []
for i in range(len(uniques)):
    print(i)
    indeces.append([])
    for j in range(len(x)):
        if np.array_equal(uniques[i],x[j]):
            print("Added")
            indeces[i].append(details[j])


f = open("../Pickles/RelocsSame","wb")
pickle.dump([indeces],f)
print(indeces[:10])
