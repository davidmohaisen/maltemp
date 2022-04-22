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



# f = open("../Pickles/RetrainingMatrices/HexdumpMahalanobisDates-trickbot","rb")
f = open("../Pickles/RetrainingMatrices/HexdumpMahalanobisDates-emotet","rb")
x_s,x_s_pca,date_s = pickle.load(f)

# pca = PCA(n_components=2)
# x_s_v = pca.fit_transform(x_s)

pca = TSNE(n_components=2,perplexity=3)
x_s_v = pca.fit_transform(x_s)

print(x_s_v.shape)

samples = []
dates = []
for i in range(len(date_s)):
    # if date_s[i] not in date_s[:i]:
        samples.append(x_s_v[i])
        dates.append(date_s[i])

counter = 1
for sample in samples:
    plt.scatter(sample[0], sample[1], marker="o", c="k", alpha = (counter/len(samples)))

    plt.text(sample[0], sample[1]+5, dates[counter-1] , horizontalalignment='center',verticalalignment='center')
    counter += 1
plt.show()
