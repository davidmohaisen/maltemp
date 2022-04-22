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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

def GetFlatten(data):
    x = []
    for i in range(len(data)):
        x.append(data[i].flatten())
    x = np.asarray(x)
    return x

def getFamilies(pathMalware, pathBenign, MalwareList, BenignList):
    Families = []
    for d in MalwareList:
        try:
            if d[6] != "SINGLETON" and d[6] != "-":
                Families.append(d[6])

        except:
            print("passed")
    Families = list(set(Families))
    return Families


def load_data_for_family(pathMalware, pathBenign, MalwareList, BenignList, FamilyName):
    x = []
    for d in MalwareList:
        try:
            if d[6] == FamilyName:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x.append(img)
        except:
            print("passed")
    x = np.asarray(x).astype('float32') / 255
    return x



def load_data_for_family_PerMonth(pathMalware, pathBenign, MalwareList, BenignList, FamilyName):
    x = []
    lastMonth = -1
    for d in MalwareList:
        try:
            if d[6] == FamilyName and d[3] != lastMonth:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x.append(img)
                lastMonth = d[3]
        except:
            print("passed")
    x = np.asarray(x).astype('float32') / 255
    return x

def GetEmbedding(model, data, layerIndex):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=model.layers[layerIndex].output)
    embeddings = extractor.predict(data)
    return embeddings


def load_data_for_family_PerMonthTwoYears(pathMalware, pathBenign, MalwareList, BenignList, FamilyName):
    x = []
    lastMonth = -1
    for d in MalwareList:
        try:
            if d[2] >= 2019 and d[6] == FamilyName and d[3] != lastMonth:
                f = open(pathMalware+d[0],"rb")
                img = pickle.load(f)
                x.append(img)
                lastMonth = d[3]
        except:
            print("passed")
    x = np.asarray(x).astype('float32') / 255
    return x

def GetEmbedding(model, data, layerIndex):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=model.layers[layerIndex].output)
    embeddings = extractor.predict(data)
    return embeddings

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


input_shape = (64,64,1)




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

model = keras.models.load_model("../Model/Image/Baseline_.039.h5")

# model.summary()


pathMalware = "../Data/Win/Images/Pickle/Malware/"
pathBenign = "../Data/Win/Images/Pickle/Benign/"
f = open("../Pickles/BenignList","rb")
BenignList = pickle.load(f)
f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


Families = getFamilies(pathMalware, pathBenign, MalwareList, BenignList)

x_all = []
colors_all = []
ConsideredFamilies = []
# colors = ["black","blue","red","green","orange","purple","gray","pink","magenta","gold"]
colors = ["black","blue","red","green","orange","purple","gray","magenta","gold"]
counter = 0
for FamilyName in Families:
    x =  load_data_for_family_PerMonthTwoYears(pathMalware, pathBenign, MalwareList, BenignList,FamilyName)
    if len(x) != 5:
        continue
    ConsideredFamilies.append(FamilyName)
    # x = np.asarray(x[:6])
    print(x.shape)
    for i in range(len(x)):
        x_all.append(x[i])
        colors_all.append(colors[counter])
    counter += 1
    if counter == 9:
        break
x_all = np.asarray(x_all)
x_all = x_all.reshape((x_all.shape[0],x_all.shape[1],x_all.shape[2],1))

print(x_all.shape)
# x_embedding = GetEmbedding(model, x_all, -2)
x_embedding = GetFlatten(x_all)
print(x_embedding.shape)

# tsne = TSNE(n_components=2, verbose=0,n_jobs=5,learning_rate=5,perplexity=140)
tsne = PCA(n_components=2)
tsne_results = tsne.fit_transform(x_embedding)

plt.figure(figsize=(6, 5))
colorAlpha = [0]*len(colors)
for i in range(len(tsne_results)):
    plt.scatter(tsne_results[i][0],tsne_results[i][1], c=colors_all[i],alpha=min(1,0.10+(colorAlpha[colors.index(colors_all[i])])/(colors_all.count(colors_all[i]))))
    colorAlpha[colors.index(colors_all[i])] += 1
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
plt.legend(lines, ConsideredFamilies)
plt.show()



# for j in range(len(colors)):
#     plt.figure(figsize=(6, 5))
#     colorAlpha = [0]*len(colors)
#     for i in range(len(tsne_results)):
#         if colors[j] == colors_all[i]:
#             # plt.scatter(tsne_results[i][0],tsne_results[i][1], c="black",alpha=min(1,0.10+(colorAlpha[colors.index(colors_all[i])]//10)/(colors_all.count(colors_all[i])//10)))
#             plt.scatter(tsne_results[i][0],tsne_results[i][1], c="black",alpha=min(1,0.10+(colorAlpha[colors.index(colors_all[i])])/(colors_all.count(colors_all[i]))))
#             colorAlpha[colors.index(colors_all[i])] += 1
#     lines = [Line2D([0], [0], color="black", linewidth=3, linestyle='-') ]
#     plt.legend(lines, [ConsideredFamilies[j]])
#     plt.show()


# x_all = []
# for FamilyName in Families:
#     x =  load_data_for_family_PerMonth(pathMalware, pathBenign, MalwareList, BenignList,FamilyName)
#     for i in range(len(x)):
#         x_all.append(x[i])
# x_all = np.asarray(x_all)
# x_all = x_all.reshape((x_all.shape[0],x_all.shape[1],x_all.shape[2],1))
#
# print(x_all.shape)
# x_embedding = GetEmbedding(model, x_all, -2)
# print(x_embedding.shape)
#
# # tsne = TSNE(n_components=2, verbose=0,n_jobs=5,learning_rate=5,perplexity=140)
# pca = PCA(n_components=2)
# pca.fit(x_embedding)
#
# x_all = []
# colors_all = []
# ConsideredFamilies = []
# # colors = ["black","blue","red","green","orange","purple","gray","pink","magenta","gold"]
# for FamilyName in Families:
#     x =  load_data_for_family_PerMonth(pathMalware, pathBenign, MalwareList, BenignList,FamilyName)
#     if len(x) > 10 or len(x) < 5 :
#         continue
#
#     print(x.shape)
#     x = np.asarray(x)
#     x = x.reshape((x.shape[0],x.shape[1],x.shape[2],1))
#     x_embedding = GetEmbedding(model, x, -2)
#     print(x_embedding.shape)
#     # pca.fit(x_embedding)
#     PCA_results = pca.transform(x_embedding)
#     plt.figure(figsize=(6, 5))
#
#     for i in range(len(PCA_results)):
#         plt.scatter(PCA_results[i][0],PCA_results[i][1], c="black",alpha=i/len(PCA_results))
#     # plt.legend()
#     lines = [Line2D([0], [0], color="black", linewidth=3, linestyle='-')]
#     plt.legend(lines, [FamilyName])
#     plt.show()
