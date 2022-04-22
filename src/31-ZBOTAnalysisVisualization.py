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
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

def removeElements(lst, k):
    counted = Counter(lst)
    return [el for el in lst if counted[el] >= k]


matricesNames = ["Hexdump","Entropy","Functions","Sections","Relocs","Strings"]


datesCommon = []


for metric in matricesNames:
    f = open("../Pickles/RetrainingMatrices/"+metric+"MahalanobisDates-zbot","rb")
    x_s,x_s_pca,date_s = pickle.load(f)
    # if datetime.strptime(date_s[0], '%m/%d/%y').strftime("%m/%d/%Y") not in datesCommon:
    if datetime.strptime(date_s[0], '%m/%d/%y').strftime("%m/%d/%y") not in datesCommon:
        # datesCommon.append(date_s[0])
        d2 = datetime.strptime(date_s[0], '%m/%d/%y')
        # datesCommon.append(d2.strftime("%m/%d/%Y"))
        datesCommon.append(d2.strftime("%m/%d/%y"))
matricesNames = ["Functions"]



for metric in matricesNames:
    f = open("../Pickles/RetrainingMatrices/"+metric+"MahalanobisDates-zbot","rb")
    x_s,x_s_pca,date_s = pickle.load(f)
    samples = []
    for i in range(len(date_s)):
        if date_s[i] not in date_s[:i]:
            if len(datesCommon)!= 0:
                d2 = datetime.strptime(date_s[i], '%m/%d/%y')
                # d1 = datetime.strptime(datesCommon[-1], '%m/%d/%Y')
                d1 = datetime.strptime(datesCommon[-1], '%m/%d/%y')
                if (d2 - d1).days > 20:
                    samples.append(x_s[i])
                    # datesCommon.append(date_s[i])
                    # datesCommon.append(d2.strftime("%m/%d/%Y"))
                    datesCommon.append(d2.strftime("%m/%d/%y"))

            else:
                samples.append(x_s[i])
                # datesCommon.append(date_s[i])
                d2 = datetime.strptime(date_s[i], '%m/%d/%y')
                # datesCommon.append(d2.strftime("%m/%d/%Y"))
                datesCommon.append(d2.strftime("%m/%d/%y"))

print("\n".join(datesCommon))






# Convert date strings (e.g. 2014-10-18) to datetime
# dates = [datetime.strptime(d, "%m/%d/%Y") for d in datesCommon]
dates = [datetime.strptime(d, "%m/%d/%y") for d in datesCommon]

# Choose some nice levels
levels = np.tile([100,-100, 75, -75, 50, -50],
                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(figsize=(18, 3))
# ax.set(title="Matplotlib release dates")

markerline, stemline, baseline = ax.stem(dates, levels,
                                         linefmt="C3-", basefmt="k-")


plt.setp(markerline, mec="k", mfc="k", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dates)))

# annotate lines
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(dates, levels, datesCommon, vert,rot,marginY):
    ax.annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=8.5)

# format xaxis with 4 month intervals
# ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=9))
# ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
# plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# remove y axis and spines
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)
plt.ylim(bottom=-300, top = 200)
ax.margins(y=0.1)
plt.show()
