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
    file = open(path+name,"rb")
    data = pickle.load(file)
    weekly = []
    biweekly = []
    monthly = []
    twoMonths = []
    threeMonths = []
    sixMonths = []
    yearly = []
    for i in range(len(data)):
        weekly.append(data[i][i])
        biweekly.append(data[i-(i%2)][i])
        monthly.append(data[i-(i%4)][i])
        twoMonths.append(data[i-(i%8)][i])
        threeMonths.append(data[i-(i%13)][i])
        sixMonths.append(data[i-(i%26)][i])
        yearly.append(data[i-(i%52)][i])
    print("=============================")
    print(name)
    print("weekly",sum(weekly)/len(weekly))
    print("bi-weekly",sum(biweekly)/len(biweekly))
    print("monthly",sum(monthly)/len(monthly))
    print("2-months",sum(twoMonths)/len(twoMonths))
    print("3-months",sum(threeMonths)/len(threeMonths))
    print("6-months",sum(sixMonths)/len(sixMonths))
    print("yearly",sum(yearly)/len(yearly))
    print("no training",sum(data[0])/len(data[0]))

    print(name,"&",round(100*sum(data[0])/len(data[0]),2),"&",round(100*sum(weekly)/len(weekly),2),"&",round(100*sum(biweekly)/len(biweekly),2),"&",round(100*sum(monthly)/len(monthly),2),"&",round(100*sum(twoMonths)/len(twoMonths),2),"&",round(100*sum(threeMonths)/len(threeMonths),2),"&",round(100*sum(sixMonths)/len(sixMonths),2),"&",round(100*sum(yearly)/len(yearly),2),"\\\\\\Xhline{1\\arrayrulewidth}")
