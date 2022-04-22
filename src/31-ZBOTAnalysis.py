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
from collections import Counter
import datetime

def removeElements(lst, k):
    counted = Counter(lst)
    return [el for el in lst if counted[el] >= k]


matricesNames = ["Hexdump","Entropy","Functions","Sections","Relocs","Strings"]


datesCommon = []


for metric in matricesNames:
    f = open("../Pickles/RetrainingMatrices/"+metric+"MahalanobisDates-zbot","rb")
    x_s,x_s_pca,date_s = pickle.load(f)
    if date_s[0] not in datesCommon:
        datesCommon.append(date_s[0])

matricesNames = ["Functions"]



for metric in matricesNames:
    f = open("../Pickles/RetrainingMatrices/"+metric+"MahalanobisDates-zbot","rb")
    x_s,x_s_pca,date_s = pickle.load(f)
    samples = []
    for i in range(len(date_s)):
        if date_s[i] not in date_s[:i]:
            if len(datesCommon)!= 0:
                d2 = datetime.datetime.strptime(date_s[i], '%m/%d/%y')
                d1 = datetime.datetime.strptime(datesCommon[-1], '%m/%d/%y')
                if (d2 - d1).days > 7:
                    samples.append(x_s[i])
                    datesCommon.append(date_s[i])
            else:
                samples.append(x_s[i])
                datesCommon.append(date_s[i])

print("\n".join(datesCommon))
# exit()
matricesNames = ["Hexdump","Entropy","Functions","Sections","Relocs","Strings"]
# matricesNames = ["Hexdump"]


for metric in matricesNames:
    f = open("../Pickles/RetrainingMatrices/"+metric+"MahalanobisDates-zbot","rb")
    x_s,x_s_pca,date_s = pickle.load(f)

    print(metric)

    samples = []
    dates = []
    for i in range(len(date_s)):
        if date_s[i] not in date_s[:i] and date_s[i] in datesCommon:
            samples.append(x_s[i])
            dates.append(date_s[i])

    pca = TSNE(n_components=2,perplexity=10,metric="cosine")
    # pca = PCA(n_components=2)
    samples = pca.fit_transform(samples)
    print(len(samples))
    counter = 1
    for sample in samples:
        plt.scatter(sample[0], sample[1], marker="o", c="k", alpha = (counter/len(samples)))

        plt.text(sample[0], sample[1], dates[counter-1] , horizontalalignment='center',verticalalignment='center')
        counter += 1
    plt.show()
