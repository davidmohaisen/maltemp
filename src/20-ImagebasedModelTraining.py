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
                f = open(pathMalware+d[0],"rb")
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
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test.append(img)
                y_test.append(1)
            else:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x_test_old.append(img)
                y_test_old.append(1)
        except:
            print("passed")
    counter = 0
    for d in range(len(BenignList)):
        if d >= (0.75*len(BenignList)):
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test.append(img)
            y_test.append(0)
        if d >= (0.50*len(BenignList)):
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            x_test_old.append(img)
            y_test_old.append(0)
        else:
            f = open(pathBenign+BenignList[d],"rb")
            img = pickle.load(f)
            if counter== 4:
                counter = 0
                x_vali.append(img)
                y_vali.append(0)
            else :
                x_train.append(img)
                y_train.append(0)
            counter += 1
    x_train = np.asarray(x_train).astype('float32') / 255
    y_train = np.asarray(y_train)
    x_vali = np.asarray(x_vali).astype('float32') / 255
    y_vali = np.asarray(y_vali)
    x_test = np.asarray(x_test).astype('float32') / 255
    y_test = np.asarray(y_test)
    x_test_old = np.asarray(x_test_old).astype('float32') / 255
    y_test_old = np.asarray(y_test_old)
    return x_train,y_train,x_vali, y_vali,x_test,y_test,x_test_old,y_test_old

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
            if d[2] >= yearStart and d[2] <= yearEnd :
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
            if d[2] >= yearStart and d[2] <= yearEnd:
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


pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"
yearStart = 2017
yearEnd = 2018
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)

f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)

x_train, y_train,x_vali, y_vali, x_test, y_test, x_test_old, y_test_old =  load_data_time_at_once(pathMalware, pathBenign, MalwareList, BenignList, yearStart,yearEnd)
print(x_train.shape)
print(y_train.shape)
print(x_vali.shape)
print(y_vali.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test_old.shape)
print(y_test_old.shape)


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
x_vali = np.reshape(x_vali,(x_vali.shape[0],x_vali.shape[1],x_vali.shape[2],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
x_test_old = np.reshape(x_test_old,(x_test_old.shape[0],x_test_old.shape[1],x_test_old.shape[2],1))
y_train = keras.utils.to_categorical(y_train, 2)
y_vali = keras.utils.to_categorical(y_vali, 2)
y_test = keras.utils.to_categorical(y_test, 2)
y_test_old = keras.utils.to_categorical(y_test_old, 2)



# Training parameters
batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 30
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
    if epoch > 20:
        lr *= 0.5e-1
    elif epoch > 10:
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
model_name = 'Updated2017_.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)


callbacks = [checkpoint, lr_reducer, lr_scheduler]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          validation_data=(x_vali, y_vali),
          callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', scores[1])
scores = model.evaluate(x_vali, y_vali, verbose=0)
print('Vali accuracy:', scores[1])
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
scores = model.evaluate(x_test_old, y_test_old, verbose=0)
print('Test Old accuracy:', scores[1])


#
model_json = model.to_json()
with open("../Model/Image/Baseline.json", "w") as json_file:
    json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../Model/Image/Baseline.h5")
# print("Saved model to disk")
# model.save("../Model/Image/Baseline.h5")
