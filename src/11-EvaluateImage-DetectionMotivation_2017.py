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
import sklearn
import datetime

def calculateEval(y_true,y_pred):
    p = 0
    tp = 0
    n = 0
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            n += 1
            if y_true[i] == y_pred[i]:
                tn += 1
        else:
            p += 1
            if y_true[i] == y_pred[i]:
                tp += 1
    tnr = 1.0*tn/n
    tpr = 1.0*tp/p
    # fscore = tp/(tp+0.5*((p-tp)+(n-tn)))
    fscore = (sklearn.metrics.f1_score(y_true,y_pred)+sklearn.metrics.f1_score(y_true,y_pred,pos_label=0))/2
    acc = (tp+tn)/(p+n)
    return acc,tnr,tpr,fscore

def load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, year):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for d in MalwareList:
        try:
            if d[2] >= year:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            else:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_train.append(img)
                y_train.append(1)
        except:
            print("passed")
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test.append(img)
            y_test.append(0)
        else:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_train.append(img)
            y_train.append(0)
    x_train = np.asarray(x_train).astype('float32') / 255
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test).astype('float32') / 255
    y_test = np.asarray(y_test)
    return x_train,y_train,x_test,y_test

def load_data_random(pathMalware, pathBenign, MalwareList, BenignList):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    malX = []
    malY = []
    for d in MalwareList:
        try:
            f = open(pathMalware+d[0],"rb")
            img = pickle.load(f)
            malX.append(img)
            malY.append(1)
        except:
            print("passed")
    x_train, x_test, y_train, y_test = train_test_split(malX, malY, test_size=0.50, random_state=42)
    x_train = list(x_train)
    x_test = list(x_test)
    y_train = list(y_train)
    y_test = list(y_test)


    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test.append(img)
            y_test.append(0)
        else:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_train.append(img)
            y_train.append(0)
    x_train = np.asarray(x_train).astype('float32') / 255
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test).astype('float32') / 255
    y_test = np.asarray(y_test)
    return x_train,y_train,x_test,y_test

def load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            index = int((d-(len(BenignList)/2))//step)
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test[index].append(img)
            y_test[index].append(0)
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test


def load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    weeks = 53
    for year in range(years):
        for week in range(weeks):
            x_test.append([])
            y_test.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*53)+d[4]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            index = int((d-(len(BenignList)/2))//step)
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test[index].append(img)
            y_test[index].append(0)
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test



def load_test_data_monthly_old_years(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd):
    x_test = []
    y_test = []
    years =  yearEnd-yearStart + 1
    months = 12
    for year in range(years):
        for month in range(months):
            x_test.append([])
            y_test.append([])

    for d in MalwareList:
        try:
            if d[2] >= yearStart and d[2] <= yearEnd:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                index = ((d[2]-yearStart)*12)+d[3]-1
                x_test[index].append(img)
                y_test[index].append(1)
        except:
            print("passed")
    step = (len(BenignList)/2)/len(x_test)
    for d in range(len(BenignList)):
        if d >= len(BenignList)/2:
            index = int((d-(len(BenignList)/2))//step)
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test[index].append(img)
            y_test[index].append(0)
    for i in range(len(x_test)):
        x_test[i] = np.asarray(x_test[i]).astype('float32') / 255
        y_test[i] = np.asarray(y_test[i])
    return x_test,y_test



pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"
yearStart = 2019
yearEnd = 2020
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train, x_test, y_test =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



# x_train, y_train, x_test, y_test =  load_data_random(pathMalware, pathBenign, MalwareList, BenignList)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# x_test, y_test =  load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
# for i in range(len(x_test)):
#     print(i,x_test[i].shape,y_test[i].shape)
    # print(collections.Counter(y_test[i]))


# x_test, y_test =  load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
# for i in range(len(x_test)):
#     print(i,x_test[i].shape,y_test[i].shape)
#     print(collections.Counter(y_test[i]))


x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.50, random_state=10)


# Training parameters
batch_size = 64  # orig paper trained all networks with batch_size=128
epochs = 50
num_classes = 2


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 6


depth = n * 9 + 2


input_shape = x_train.shape[1:]


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)




def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v2(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
# model.summary()



lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), '../Model/Image/')
model_name = 'Baseline_.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)


callbacks = [checkpoint, lr_reducer, lr_scheduler]

model = keras.models.load_model("../Model/Image/Updated2017_.014.h5")



# y_preds = model.predict(x_train)
# AUC = sklearn.metrics.roc_auc_score(y_train,y_preds)
# print("Train AUC", AUC)
# y_true = np.argmax(y_train, axis = 1)
# y_preds = np.argmax(model.predict(x_train), axis = 1)
# acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
# print("Train: ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)

# y_preds = model.predict(x_vali)
# AUC = sklearn.metrics.roc_auc_score(y_vali,y_preds)
# print("Vali AUC", AUC)
# y_true = np.argmax(y_vali, axis = 1)
# y_preds = np.argmax(model.predict(x_vali), axis = 1)
# acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
# print("Vali: ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)

# y_preds = model.predict(x_test)
# AUC = sklearn.metrics.roc_auc_score(y_test,y_preds)
# print("Test AUC", AUC)
# y_true = np.argmax(y_test, axis = 1)
# y_preds = np.argmax(model.predict(x_test), axis = 1)
# acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
# print("Test: ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)

# x_test, y_test =  load_test_data_monthly_old_years(pathMalware, pathBenign, MalwareList, BenignList, 2017, 2018)
# tprMonthly = []
# LabelString = []
# indexLabel = []
# for i in range(len(x_test)):
#     if 1 not in collections.Counter(y_test[i]).keys():
#         continue
#     if i%3 == 0:
#         indexLabel.append(i)
#         yearL = 2019+(i//12)
#         WeekL = (i%12)+1
#         d = str(yearL)+'-'+str(WeekL)
#         r = datetime.datetime.strptime(d, "%Y-%m")
#         LabelString.append(r.strftime("%b %y"))
#
#     x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2],1))
#     y_test[i] = keras.utils.to_categorical(y_test[i], 2)
#     y_true = np.argmax(y_test[i], axis = 1)
#     y_preds = np.argmax(model.predict(x_test[i]), axis = 1)
#     acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
#     print("Month:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
#     tprMonthly.append(tpr)
#
# plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-ok');
# z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 1)
# p = np.poly1d(z)
# plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)
# plt.xticks(labels = LabelString, ticks = indexLabel)
# plt.xlabel("Time", fontsize=18)
# plt.ylabel("Performance", fontsize=18)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
# plt.ylim(0.60, 1.00)
#
# plt.show()
#
#
# x_test, y_test =  load_test_data_monthly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
# tprMonthly = []
# LabelString = []
# indexLabel = []
# for i in range(len(x_test)):
#     if 1 not in collections.Counter(y_test[i]).keys():
#         continue
#     if i%3 == 0:
#         indexLabel.append(i)
#         yearL = 2019+(i//12)
#         WeekL = (i%12)+1
#         d = str(yearL)+'-'+str(WeekL)
#         r = datetime.datetime.strptime(d, "%Y-%m")
#         LabelString.append(r.strftime("%b %y"))
#
#     x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2],1))
#     y_test[i] = keras.utils.to_categorical(y_test[i], 2)
#     y_true = np.argmax(y_test[i], axis = 1)
#     y_preds = np.argmax(model.predict(x_test[i]), axis = 1)
#     acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
#     print("Month:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
#     tprMonthly.append(tpr)
#
# plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-ok');
# z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 1)
# p = np.poly1d(z)
# plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)
# plt.xticks(labels = LabelString, ticks = indexLabel)
# plt.xlabel("Time", fontsize=18)
# plt.ylabel("Performance", fontsize=18)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
# plt.ylim(0.60, 1.00)
#
# plt.show()
#


x_test, y_test =  load_test_data_monthly_old_years(pathMalware, pathBenign, MalwareList, BenignList, 2013, 2020)
tprMonthly = []
LabelString = []
indexLabel = []
for i in range(len(x_test)):
    if 1 not in collections.Counter(y_test[i]).keys():
        continue
    if i%12 == 0:
        indexLabel.append(i)
        yearL = 2013+(i//12)
        WeekL = (i%12)+1
        d = str(yearL)+'-'+str(WeekL)
        r = datetime.datetime.strptime(d, "%Y-%m")
        # LabelString.append(r.strftime("%b %y"))
        LabelString.append(r.strftime("%y"))

    x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2],1))
    y_test[i] = keras.utils.to_categorical(y_test[i], 2)
    y_true = np.argmax(y_test[i], axis = 1)
    y_preds = np.argmax(model.predict(x_test[i]), axis = 1)
    acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
    print("Month:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
    tprMonthly.append(tpr)

plt.plot(np.arange(0,len(tprMonthly)), tprMonthly, '-ok');
z = np.polyfit(np.arange(0,len(tprMonthly)), tprMonthly, 3)
p = np.poly1d(z)
plt.plot(np.arange(0,len(tprMonthly)),p(np.arange(0,len(tprMonthly))),"k--",linewidth=2)
plt.xticks(labels = LabelString, ticks = indexLabel)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Performance", fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.00, 1.00)

plt.show()


# x_test, y_test =  load_test_data_weekly(pathMalware, pathBenign, MalwareList, BenignList, yearStart, yearEnd)
# tprWeekly = []
# LabelString = []
# indexLabel = []
# for i in range(len(x_test)):
#     if 1 not in collections.Counter(y_test[i]).keys():
#         continue
#     if i%15 == 0:
#         indexLabel.append(i)
#         yearL = 2019+(i//53)
#         WeekL = (i%53)+1
#         d = str(yearL)+'-W'+str(WeekL)
#         # r = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
#         r = datetime.datetime.strptime(d+"-1", "%Y-W%W-%w")
#
#         LabelString.append(r.strftime("%b %y"))
#     x_test[i] = np.reshape(x_test[i],(x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2],1))
#     y_test[i] = keras.utils.to_categorical(y_test[i], 2)
#     y_true = np.argmax(y_test[i], axis = 1)
#     y_preds = np.argmax(model.predict(x_test[i]), axis = 1)
#     acc,tnr,tpr,fscore = calculateEval(y_true,y_preds)
#     print("Week:",i,"ACC",acc,"TNR",tnr,"TPR",tpr,"F-1 score",fscore)
#     tprWeekly.append(tpr)
#
# plt.plot(np.arange(0,len(tprWeekly)), tprWeekly, '-ok');
# z = np.polyfit(np.arange(0,len(tprWeekly)), tprWeekly, 1)
# p = np.poly1d(z)
# plt.plot(np.arange(0,len(tprWeekly)),p(np.arange(0,len(tprWeekly))),"k--",linewidth=2)
#
# plt.xticks(labels = LabelString, ticks = indexLabel)
# plt.xlabel("Time", fontsize=18)
# plt.ylabel("Performance", fontsize=18)
#
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
# plt.ylim(0.60, 1.00)
#
# plt.show()
