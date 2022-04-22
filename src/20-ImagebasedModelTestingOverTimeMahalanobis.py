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
from sklearn.neighbors import LocalOutlierFactor

def load_test_data_weekly(pathMalware, MalwareList, fam):
    x_test = []
    date = []

    for d in MalwareList:
        try:
            if  d[6] == fam:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test.append(img)
                date.append(d[5].split(" ")[0])
        except:
            print("passed")
    x_test = np.asarray(x_test).astype('float32') / 255
    date = np.asarray(date)
    return x_test,date



pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"


f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


x_test, dates =  load_test_data_weekly(pathMalware, MalwareList, fam = "trickbot")



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
    C = np.sqrt(chi2.ppf((1-0.000001), df=df.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


x_s = []
x_s_pca = []
date_s = []


x_test_new = []
for xsample in x_test:
    x_test_new.append(resize(xsample, (16, 16)))

x_test_new = np.asarray(x_test_new)

print(x_test_new.shape)

x_s.append(x_test[0])
x_s_pca.append(x_test_new[0])
date_s.append(dates[0])

robust_mean, inv_covmat = initiate_mahalanobis(np.reshape(x_test_new[:2],(x_test_new[:2].shape[0],x_test_new[:2].shape[1]*x_test_new[:2].shape[2])))


for i in range(len(x_test_new)):
    print(i,len(x_test_new))


    outliers, md_rb = calculate_mahalanobis(np.reshape(x_test_new[i:i+1],(x_test_new[i:i+1].shape[0],x_test_new[i:i+1].shape[1]*x_test_new[i:i+1].shape[2])),robust_mean, inv_covmat)


    if len(outliers) >= 1:
        robust_mean, inv_covmat = initiate_mahalanobis(np.reshape(x_test_new[:i+1],(x_test_new[:i+1].shape[0],x_test_new[:i+1].shape[1]*x_test_new[:i+1].shape[2])))
        x_s.append(x_test[i])
        x_s_pca.append(x_test_new[i])
        date_s.append(dates[i])

x_s = np.asarray(x_s)
x_s_pca = np.asarray(x_s_pca)
date_s = np.asarray(date_s)


f = open("../Pickles/RetrainingMatrices/ImageMahalanobisDatesTrickbot","wb")
pickle.dump([x_s,x_s_pca,date_s],f)

print(x_s.shape,x_s_pca.shape)
print(date_s)
