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

matricesNames = ["Image","Hexdump","Entropy","Functions","Sections","Relocs","Strings"]
path = "../Pickles/RetrainingMatrices/"

for name in matricesNames:
    file = open(path+name+"ConfAVG","rb")
    _,When = pickle.load(file)
    print("=============================")
    print(name)
    results = []
    file = open(path+name,"rb")
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

    results.append(str(round(100.0*sum(currentR)/len(currentR),2)))
    print(TimesRetrain)


    file = open(path+name+"SplitData","rb")
    data = pickle.load(file)

    for d in data:
        currentI = 0
        pointerI = 0
        currentR = []
        counter = 0
        for i in range(len(d)):
            currentR.append(d[currentI][i])
            if (pointerI < len(When) and i == When[pointerI]) or (counter != 0 and (counter%13) == 0):
                pointerI += 1
                currentI = i
                counter = 0
            else:
                counter += 1
        results.append(str(round(100.0*sum(currentR)/len(currentR),2)))
    print(" & ".join(results)+" & ")
