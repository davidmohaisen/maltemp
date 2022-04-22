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


def load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, year, endYear):
    x_train = []
    y_train = []
    x_vali = []
    y_vali = []
    x_test = []
    y_test = []
    x_test_old = []
    y_test_old = []
    counter = 0
    for d in MalwareList:
        try:
            if d[2] >= year and  d[2] <= endYear:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                if counter== 4:
                    counter = 0
                    x_vali.append(img)
                    y_vali.append(1)
                else :
                    x_train.append(img)
                    y_train.append(1)
                counter += 1
            elif d[2] >= endYear:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            else:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(1)
        except:
            print("passed")
    counter = 0
    for d in range(len(BenignList)):
        try:
            if d >= (0.75*len(BenignList)):
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(0)
            if d >= (0.50*len(BenignList)):
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(0)
            else:
                f = open(pathBenign+BenignList[d]+".pickle","rb")
                img = pickle.load(f)
                if counter== 4:
                    counter = 0
                    x_vali.append(img)
                    y_vali.append(0)
                else :
                    x_train.append(img)
                    y_train.append(0)
                counter += 1
        except:
            print("passed")
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_vali = np.asarray(x_vali)
    y_vali = np.asarray(y_vali)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_test_old = np.asarray(x_test_old)
    y_test_old = np.asarray(y_test_old)
    return x_train,y_train,x_vali, y_vali,x_test,y_test,x_test_old,y_test_old

def load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020):
    x_test = []
    y_test = []
    Fam = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])
            Fam.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd:
                f = open(pathMalware+d[0]+".pickle","rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
                Fam[index].append(d[6])
        except:
            print("passed")

    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i])
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test,Fam





pathMalware = "../Data/Win/StaticReverseEngineer/FilteredMalware/hexdump/"
pathBenign = "../Data/Win/StaticReverseEngineer/Benign/hexdump/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train,x_vali, y_vali, x_test, y_test, x_test_old, y_test_old =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)



x_test, y_test,Fam =  load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart=2019,yearEnd=2020)
x_test = x_test[:92]
y_test = y_test[:92]
Fam = Fam[:92]



import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
import scipy as sp
from sklearn.decomposition import PCA
from skimage.transform import resize
from sklearn.manifold import TSNE


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
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
            thresholds.append(value)
        else:
            continue
    return outlier, thresholds, md


def getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, yearstart,yearend):
    Families = []
    for d in MalwareList:
        try:
            if d[2] >= yearstart and  d[2] <= yearend and d[6] != "SINGLETON":
                Families.append(d[6])

        except:
            print("passed")
    Families = list(set(Families))
    return Families



TrainFamilies = getTrainFamilies(pathMalware, pathBenign, MalwareList, BenignList, 2017,2018)



RetrainTime = []
Performance = []

f = open("../Pickles/RetrainingMatrices/Hexdump","rb")
matrixToCal = pickle.load(f)

currentI = 0


print(x_train.shape)

pca = PCA(n_components=200)
x_train_pca = pca.fit_transform(x_train)
print(x_train_pca.shape)
robust_mean, inv_covmat = initiate_mahalanobis(x_train_pca)


FamAnalysis_seen = []
FamAnalysis_seen_color = []
FamAnalysis_seen_shape = []
FamAnalysis_unseen = []
FamAnalysis_unseen_color = []
FamAnalysis_unseen_shape = []
FamAnalysis_sing = []
FamAnalysis_sing_color = []
FamAnalysis_sing_shape = []

currentNewFam  = []

for i in range(len(x_test)):
    print(i,len(x_test))
    if len(x_test[i]) == 0:
        Performance.append(Performance[-1])
        continue

    x_test_n = x_test[i]
    x_test_n = pca.transform(x_test_n)


    Performance.append(matrixToCal[currentI][i])



    x_train_pca = np.concatenate((x_train_pca,x_test_n))
    y_train = np.concatenate((y_train,y_test[i]))


    outliers,thresholds, md_rb = calculate_mahalanobis(x_test_n,robust_mean, inv_covmat)



    for j in range(len(Fam[i])):
        if j in outliers:
            if Fam[i][j] in TrainFamilies:
                FamAnalysis_seen.append(x_test_n[j])
                FamAnalysis_seen_color.append("r")
                FamAnalysis_seen_shape.append("v")
            elif Fam[i][j]  == "SINGLETON":
                FamAnalysis_sing.append(x_test_n[j])
                FamAnalysis_sing_color.append("r")
                FamAnalysis_sing_shape.append("P")
            else:
                FamAnalysis_unseen.append(x_test_n[j])
                FamAnalysis_unseen_color.append("r")
                FamAnalysis_unseen_shape.append("s")
                currentNewFam.append(Fam[i][j])
        else:
            if Fam[i][j] in TrainFamilies:
                FamAnalysis_seen.append(x_test_n[j])
                FamAnalysis_seen_color.append("k")
                FamAnalysis_seen_shape.append("v")
            elif Fam[i][j]  == "SINGLETON":
                FamAnalysis_sing.append(x_test_n[j])
                FamAnalysis_sing_color.append("k")
                FamAnalysis_sing_shape.append("P")
            else:
                FamAnalysis_unseen.append(x_test_n[j])
                FamAnalysis_unseen_color.append("k")
                FamAnalysis_unseen_shape.append("s")
                currentNewFam.append(Fam[i][j])

    if len(outliers) >= 1:
        currentI = i
        RetrainTime.append(i)
        robust_mean, inv_covmat = initiate_mahalanobis(x_train_pca)
        TrainFamilies = list(set(TrainFamilies+currentNewFam))
        currentNewFam = []


print(len(FamAnalysis_seen),len(FamAnalysis_sing),len(FamAnalysis_unseen))
print(FamAnalysis_seen_color.count("r"),FamAnalysis_sing_color.count("r"),FamAnalysis_unseen_color.count("r"))
samples = np.concatenate((FamAnalysis_seen,FamAnalysis_sing,FamAnalysis_unseen))
colors = np.concatenate((FamAnalysis_seen_color,FamAnalysis_sing_color,FamAnalysis_unseen_color))
shapes = np.concatenate((FamAnalysis_seen_shape,FamAnalysis_sing_shape,FamAnalysis_unseen_shape))

samples = TSNE(n_components=2).fit_transform(samples)

for i in range(len(samples)):
    plt.scatter(samples[i][0], samples[i][1], marker=shapes[i], c=colors[i])

plt.show()