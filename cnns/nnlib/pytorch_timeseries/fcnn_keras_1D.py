#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""

from __future__ import print_function

import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.utils import np_utils


config = tf.ConfigProto(device_count={'GPU': 4, 'CPU': 16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)

data_folder = "TimeSeriesDatasets"
ucr_path = os.path.join(dir_path, os.pardir, data_folder)

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


nb_epochs = 2000
# 'Adiac',
# flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso',
#          'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
#          'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour',
#          'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
#          'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT',
#          'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
#          'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf',
#          'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves',
#          'SwedishLeaf', 'Symbols',
#          'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns',
#          'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
#          'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']
# flist = ['50words']
flist = os.listdir(ucr_path)
flist = sorted(flist)

for each in flist:
    print("Dataset: ", each)
    fname = each
    x_train, y_train = readucr(fname, data_type="TRAIN")
    x_test, y_test = readucr(fname, data_type="TEST")
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] // 10, 16)

    y_train = ((y_train - y_train.min()) / (y_train.max() - y_train.min()) * (
            nb_classes - 1)).astype(int)
    y_test = ((y_test - y_test.min()) / (y_test.max() - y_test.min()) * (
            nb_classes - 1)).astype(int)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) // (x_train_std)

    x_test = (x_test - x_train_mean) // (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    x = keras.layers.Input(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv1D(128, 8, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv1D(256, 5, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv1D(128, 3, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.pooling.GlobalAveragePooling1D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)

    model = Model(input=x, output=out)

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
    print(log.loc[log['loss'].idxmin]['loss'],
          log.loc[log['loss'].idxmin]['val_acc'])
