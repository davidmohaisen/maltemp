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
import warnings
warnings.filterwarnings("ignore")

def load_test_data_weekly(pathMalware, MalwareList, fam):
    x_test = []
    date = []

    for d in MalwareList:
        try:
            if  d[6] == fam:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                date.append(d[5].split(" ")[0])
        except:
            print("passed")
    x_test = np.asarray(x_test)
    date = np.asarray(date)
    return x_test,date




pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/hexdump/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/hexdump/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)




x_test, dates =  load_test_data_weekly(pathMalware, MalwareList, fam = "zbot")


import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
import scipy as sp
from sklearn.decomposition import PCA
from skimage.transform import resize



def initiate_mahalanobis(df):
    #Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=512)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = sp.linalg.inv(mcd) #inverse covariance metric

    return robust_mean, inv_covmat

def calculate_mahalanobis(df,robust_mean, inv_covmat):

    #Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    #Flag as outlier
    outlier = []
    thresholds = []
    C = np.sqrt(chi2.ppf((1-0.0000001), df=df.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
            thresholds.append(value)
        else:
            continue
    return outlier, md


x_s = []
x_s_pca = []
date_s = []



pca = PCA(n_components=0.999)
x_test_pca = pca.fit_transform(x_test)
print(x_test_pca.shape)


x_s.append(x_test[0])
x_s_pca.append(x_test_pca[0])
date_s.append(dates[0])


robust_mean, inv_covmat = initiate_mahalanobis(x_test_pca[:10])


for i in range(len(x_test)):
    if i <= 10:
        continue
    print(i,len(x_test), len(date_s))

    outliers, md_rb = calculate_mahalanobis(x_test_pca[i:i+1],robust_mean, inv_covmat)


    if len(outliers) >= 1:
        robust_mean, inv_covmat = initiate_mahalanobis(x_test_pca[:i+1])
        x_s.append(x_test[i])
        x_s_pca.append(x_test_pca[i])
        date_s.append(dates[i])

x_s = np.asarray(x_s)
x_s_pca = np.asarray(x_s_pca)
date_s = np.asarray(date_s)


f = open("../Pickles/RetrainingMatrices/HexdumpMahalanobisDates-zbot","wb")
pickle.dump([x_s,x_s_pca,date_s],f)

print(x_s.shape,x_s_pca.shape)
print(date_s)
