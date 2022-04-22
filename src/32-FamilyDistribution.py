

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
from collections import Counter


def getTrainFamilies(pathMalware, MalwareList):
    Families = []
    for d in MalwareList:
        try:
            Families.append(d[6])
        except:
            print("passed")
    counter = Counter(Families)
    return counter


pathMalware = "../Data/Win/Images/Pickle/Malware/"

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


print(len(MalwareList))
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)
print(len(BenignList))

c = getTrainFamilies(pathMalware, MalwareList)
print(c)
