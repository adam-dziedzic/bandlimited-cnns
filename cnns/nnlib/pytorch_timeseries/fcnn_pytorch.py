#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
@author: ady@uchicago.edu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)


def readucr(filename, data_type):
    folder = "TimeSeriesDatasets"
    parent_path = os.path.split(os.path.abspath(dir_path))[0]
    print("parent path: ", parent_path)
    filepath = os.path.join(parent_path, folder, filename,
                            filename + "_" + data_type)
    print("filepath: ", filepath)
    data = np.loadtxt(filepath, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def getModel():
    model = nn.Sequential(
        # H' = 1 + (H + 2 * pad - HH) / stride: H' = 1 + (32 + 2 - 3) / 1 = 32
        nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=32),
        # 32 x 32
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        # np.int(((W - pool_width) // stride) + 1): (32 - 2) // 2 + 1 = 16
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(16384, 1024),  # 16384=64*16*16 input size
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024, 10),
    )

    x = keras.layers.Input(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)

    model = Model(input=x, output=out)


nb_epochs = 2000

# flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
# 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
# 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
# 'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
# 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist = ['Adiac']
for each in flist:
    fname = each
    x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] / 10, 16)

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (
            nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (
            nb_classes - 1)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    x_test = (x_test - x_train_mean) / (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1, 1,))
    x_test = x_test.reshape(x_test.shape + (1, 1,))

    model = getModel()

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=50, min_lr=0.0001)
    hist = model.fit(x_train, Y_train, batch_size=batch_size,
                     nb_epoch=nb_epochs,
                     verbose=1, validation_data=(x_test, Y_test),
                     callbacks=[reduce_lr])
    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
print
log.loc[log[‘loss
'].idxmin]['
loss’], log.loc[log[‘loss
'].idxmin][‘val_acc’]
