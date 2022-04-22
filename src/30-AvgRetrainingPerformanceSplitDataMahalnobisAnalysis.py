

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
import numpy as np
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

matricesNames = ["Image","Hexdump","Entropy","Functions","Sections","Relocs","Strings"]
path = "../Pickles/RetrainingMatrices/"
for name in matricesNames:
    # file = open(path+name+"MahalanobisAnalysis","rb")
    file = open(path+name+"MahalanobisAnalysisAllSamples","rb")

    seen,unseen,sing = pickle.load(file)
    print("=============================")
    print(name)
    print(len(unseen)-150)
    mx = max(seen+unseen+sing)
    mn = min(seen+unseen+sing)
    for i in range(len(seen)):
        seen[i] = (seen[i]-mn)/(mx-mn)
    for i in range(len(unseen)):
        unseen[i] = (unseen[i]-mn)/(mx-mn)
    for i in range(len(sing)):
        sing[i] = (sing[i]-mn)/(mx-mn)

    count, bins_count = np.histogram(seen, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf,"k-",linewidth=2, label="Seen")

    count, bins_count = np.histogram(unseen, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf,"r-",linewidth=2, label="Unseen")

    count, bins_count = np.histogram(sing, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf,"k--",linewidth=2, label="Singleton")

    plt.xlabel("Distance", fontsize=18)
    plt.ylabel("CDF", fontsize=18)
    plt.xticks(fontsize= 14)
    plt.yticks(fontsize= 14)
    plt.ylim(-0.001, 1.001)
    plt.xlim(-0.0000001, 1.0001)
    # plt.xlim(0, 126)
    plt.legend(loc='lower right',  fontsize=14)

    plt.show()
