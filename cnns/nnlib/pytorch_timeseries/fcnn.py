#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
@author: ady@uchicago.edu
"""

import argparse
import os
import socket

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateauKeras
from keras.models import Model
from keras.utils import np_utils
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import get_log_time

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)

data_folder = "TimeSeriesDatasets"
results_folder = "results"
ucr_path = os.path.join(dir_path, os.pardir, data_folder)

num_epochs = 2000

flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso',
         'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
         'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour',
         'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
         'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT',
         'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
         'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf',
         'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves',
         'SwedishLeaf', 'Symbols',
         'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns',
         'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
         'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']
# flist = ['Adiac', 'synthetic_control', "Coffee"]

# switch backend to be able to save the graphic files on the servers
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--iterations", default=1, type=int,
                    help="number of iterations for the training")
parser.add_argument("-i", "--initbatchsize", default=256, type=int,
                    help="the initial size of the batch (number of data points "
                         "for a single forward and batch passes")
parser.add_argument("-m", "--maxbatchsize", default=256, type=int,
                    help="the max size of the batch (number of data points for "
                         "a single forward and batch passes")
parser.add_argument("-s", "--startsize", default=32, type=int,
                    help="the start size of the input image")
parser.add_argument("-e", "--endsize", default=32, type=int,
                    help="the end size of the input image")
parser.add_argument("-w", "--workers", default=4, type=int,
                    help="number of workers to fetch data for pytorch data "
                         "loader, 0 means that the data will be "
                         "loaded in the main process")
parser.add_argument("-d", "--device", default="cpu",
                    help="the type of device, e.g.: cpu, cuda, cuda:0, cuda:1, "
                         "etc.")
parser.add_argument("-n", "--net", default="dense",
                    help="the type of net: alex, dense, res.")
parser.add_argument("-l", "--limit_size", default=256, type=int,
                    help="limit_size for the input for debug")
parser.add_argument("-p", "--num_epochs", default=num_epochs, type=int,
                    help="number of epochs")
parser.add_argument("-b", "--mem_test", default=False, type=bool,
                    help="is it the memory test")
parser.add_argument("-a", "--is_data_augmentation", default=True, type=bool,
                    help="should the data augmentation be applied")
parser.add_argument("-g", "--is_debug", default=False, type=bool,
                    help="is it the debug mode execution")
parser.add_argument("-c", "--conv_type", default="STANDARD",
                    help="the type of convoltution, SPECTRAL_PARAM is with the "
                         "convolutional weights initialized in the spectral "
                         "domain, please choose from: " + ",".join(
                        ConvType.get_names()))
parser.add_argument("-o", "--optimizer_type", default="ADAM",
                    help="the type of the optimizer, please choose from: " +
                         ",".join(OptimizerType.get_names()))

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

if torch.cuda.is_available():
    print("cuda is available: ")
    device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")


def getModelKeras(input_size, num_classes):
    """
    Create model.

    :param input_size: the length (width) of the time series.
    :param num_classes: number of classes
    :return: the keras model.
    """
    x = keras.layers.Input(input_size)
    conv1 = keras.layers.Conv1D(128, 8, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    conv2 = keras.layers.Conv1D(256, 5, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv3 = keras.layers.Conv1D(128, 3, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    full = keras.layers.pooling.GlobalAveragePooling1D()(conv3)
    out = keras.layers.Dense(num_classes, activation='softmax')(full)
    model = Model(input=x, output=out)
    return model


class FCNNPytorch(nn.Module):

    def __init__(self, input_size, num_clasess, kernel_sizes=[8, 5, 3],
                 out_channels=[128, 256, 128],
                 strides=[1, 1, 1]):
        """
        Create the FCNN model in PyTorch.

        :param input_size: the length (width) of the time series.
        :param num_clasess: number of output classes.
        :param kernel_sizes: the sizes of the kernesl in each conv layer.
        :param out_channels: the number of filters for each conv layer.
        :param stride: the stride for the convolutions.
        """
        super(FCNNPytorch, self).__init__()
        self.input_size = input_size
        self.num_classes = num_clasess
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.strides = strides

        self.relu = nn.ReLU(inplace=True)
        # For the "same" mode for the convolution, pad the input.
        conv_pads = [kernel_size - 1 for kernel_size in kernel_sizes]

        self.conv0 = nn.Conv1d(
            in_channels=1, out_channels=out_channels[0], stride=strides[0],
            kernel_size=kernel_sizes[0], padding=(conv_pads[0] // 2))
        self.bn0 = nn.BatchNorm1d(num_features=out_channels[0])

        self.conv1 = nn.Conv1d(
            in_channels=out_channels[0], out_channels=out_channels[1],
            kernel_size=kernel_sizes[1], padding=(conv_pads[1] // 2),
            stride=strides[1])
        self.bn1 = nn.BatchNorm1d(num_features=out_channels[1])

        self.conv2 = nn.Conv1d(
            in_channels=out_channels[1], out_channels=out_channels[2],
            kernel_size=kernel_sizes[2], padding=(conv_pads[2] // 2),
            stride=strides[2])
        self.bn2 = nn.BatchNorm1d(num_features=out_channels[2])

        self.lin = nn.Linear(input_size, num_clasess)

    def forward(self, x):
        """
        The forward pass through the network.

        :param x: the input data for the network.
        :return: the output class.
        """

        out = x

        # 0th layer.
        if self.kernel_sizes[0] % 2 == 0:
            # If kernel size is even, add one more padding value on the right.
            out = F.pad(out, (0, 1), "constant", 0)
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu(out)

        # 1st layer.
        if self.kernel_sizes[1] % 2 == 0:
            # If kernel size is even, add one more padding value on the right.
            out = F.pad(out, (0, 1), "constant", 0)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 2nd layer.
        if self.kernel_sizes[2] % 2 == 0:
            # If kernel size is even, add one more padding value on the right.
            out = F.pad(out, (0, 1), "constant", 0)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Classification.
        # Average across the channels.
        # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
        out = torch.mean(out, dim=1)
        out = self.lin(out)
        out = log_softmax(out, dim=-1)

        return out


def getModelPyTorch(input_size, num_classes):
    """
    Get the PyTorch version of the FCNN model.

    :param input_size: the length (width) of the time series.
    :param num_classes: number of output classes.

    :return: the model.
    """
    return FCNNPytorch(input_size=input_size, num_clasess=num_classes)


def readucr(filename, data_type):
    parent_path = os.path.split(os.path.abspath(dir_path))[0]
    print("parent path: ", parent_path)
    filepath = os.path.join(parent_path, data_folder, filename,
                            filename + "_" + data_type)
    print("filepath: ", filepath)
    data = np.loadtxt(filepath, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def getData(fname, normalize=True):
    x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fname + '/' + fname + '_TEST')
    num_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] // 10, 16)

    if normalize is True:
        y_train = (y_train - y_train.min()) // (
                y_train.max() - y_train.min()) * (
                          num_classes - 1)
        y_test = (y_test - y_test.min()) // (y_test.max() - y_test.min()) * (
                num_classes - 1)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean) // (x_train_std)

        x_test = (x_test - x_train_mean) // (x_train_std)

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test, batch_size, num_classes


class ToTensor(object):
    """Transform the numpy array to a tensor."""

    def __call__(self, input):
        """
        :param input: numpy array.
        :return: PyTorch's tensor.
        """
        return torch.tensor(input, dtype=torch.float)


class AddChannel(object):
    """Add channel dimension to the input time series."""

    def __call__(self, input):
        """
        Rescale the channels.

        :param image: the input image
        :return: rescaled image with the required number of channels:return:
        """
        # We receive only a single array of values as input, so have to add the
        # channel as the zero-th dimension.
        return torch.unsqueeze(input, dim=0)


class UCRDataset(Dataset):
    """One of the time-series datasets from the UCR archive."""

    def __init__(
            self, dataset_name, transformations=transforms.Compose(
                [ToTensor(), AddChannel()]), train=True):
        """
        :param dataset_name: the name of the dataset to fetch from file on disk.
        :param
        :param transformations: pytorch transforms for transforms and tensor conversion.
        """

        if train is True:
            suffix = "_TRAIN"
        else:
            suffix = "_TEST"
        csv_path = os.path.join(ucr_path, dataset_name, dataset_name + suffix)
        self.data = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        # Start class numbering from 0.
        self.labels -= self.labels.min()
        self.width = len(self.data.iloc[0]) - 1
        self.num_classes = len(np.unique(self.labels))
        self.transformations = transformations

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, val):
        self.__width = val

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, val):
        self.__num_classes = val

    def __getitem__(self, index):
        label = self.labels[index]
        # Take the row index and all values starting from the second column.
        input = np.asarray(self.data.iloc[index][1:])
        # Transform time-series input to tensor.
        if self.transformations is not None:
            input = self.transformations(input)
        # Return the time-series and the label.
        return input, label

    def __len__(self):
        # self.data.index - The index(row labels) of the DataFrame.
        return len(self.data.index)


def run_keras():
    for each in flist:
        fname = each

        x_train, y_train, x_test, y_test, batch_size, num_classes = getData(
            fname=fname, normalize=True)

        Y_train = np_utils.to_categorical(y_train, num_classes)
        Y_test = np_utils.to_categorical(y_test, num_classes)

        model = getModelKeras(input_size=x_train.shape[1:],
                              num_classes=num_classes)

        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateauKeras(monitor='loss', factor=0.5,
                                           patience=50, min_lr=0.0001)

        hist = model.fit(x_train, Y_train, batch_size=batch_size,
                         nb_epoch=num_epochs,
                         verbose=1, validation_data=(x_test, Y_test),
                         callbacks=[reduce_lr])

        # Print the testing results which has the lowest training loss.
        # Print the testing results which has the lowest training loss.
        log = pd.DataFrame(hist.history)
        print(log.loc[log['loss'].idxmin]['loss'],
              log.loc[log['loss'].idxmin]['val_acc'])


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # The cross entropy loss combines `log_softmax` and `nll_loss` in
        # a single function.
        # loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, dataset_type="test"):
    """

    :param model: deep learning model.
    :param device: cpu or gpu.
    :param test_loader: the input data.
    :param datatest_type: test or train
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        dataset_type, test_loss, correct, len(test_loader.dataset),
        accuracy))
    return test_loss, accuracy


def main(dataset_name):
    """
    The main training.

    :param dataset_name: the name of the dataset from UCR.
    """
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TimeSeries')
    min_batch_size = 16
    parser.add_argument('--min-batch-size', type=int, default=min_batch_size,
                        metavar='N',
                        help='input batch size for training (default: {})'.format(
                            min_batch_size))
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=num_epochs, metavar='N',
                        help='number of epochs to train (default: {})'.format(
                            num_epochs))
    learning_rate = 0.001
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate (default: {})'.format(
                            learning_rate))
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training '
                             'status')
    parser.add_argument("--optimizer_type", default="ADAM",
                        help="the type of the optimizer, please choose from: " +
                             ",".join(OptimizerType.get_names()))
    args = parser.parse_args()

    dataset_log_file = os.path.join(
        results_folder, get_log_time() + "-" + dataset_name + "-fcnn.log")
    with open(dataset_log_file, "a") as file:
        # Write the metadata.
        file.write("dataset," + str(dataset_name) + ",hostname," + str(
            hostname) + ",timestamp," + get_log_time() + ",num_epochs," + str(
            args.epochs) + "\n")
        # Write the header.
        file.write("epoch,train_loss,train_accuracy,test_loss,test_accuracy\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer_type = OptimizerType[args.optimizer_type]

    num_workers = 1
    pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {}

    train_dataset = UCRDataset(dataset_name, train=True)
    batch_size = min(len(train_dataset) // 10, args.min_batch_size)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        **kwargs)

    test_dataset = UCRDataset(dataset_name, train=False)
    num_classes = test_dataset.num_classes
    width = test_dataset.width
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True,
        **kwargs)

    model = getModelPyTorch(input_size=width,
                            num_classes=num_classes)
    model.to(device)

    if optimizer_type is OptimizerType.MOMENTUM:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = ReduceLROnPlateauPyTorch(optimizer=optimizer, mode='min',
                                         factor=0.5, patience=50)

    train_loss = train_accuracy = test_loss = test_accuracy = None
    for epoch in range(1, args.epochs + 1):
        train(model=model, device=device, train_loader=train_loader,
              optimizer=optimizer, epoch=epoch)
        train_loss, train_accuracy = test(model=model, device=device,
                                          test_loader=train_loader,
                                          dataset_type="train")
        test_loss, test_accuracy = test(model=model, device=device,
                                        test_loader=test_loader,
                                        dataset_type="test")
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        scheduler.step(train_loss)

        with open(dataset_log_file, "a") as file:
            file.write(str(epoch) + "," + str(train_loss) + "," + str(
                train_accuracy) + "," + str(test_loss) + "," + str(
                test_accuracy) + "\n")

    with open(global_log_file, "a") as file:
        file.write(dataset_name + "," + str(train_loss) + "," + str(
            train_accuracy) + "," + str(test_loss) + "," + str(
            test_accuracy) + "\n")


if __name__ == '__main__':
    hostname = socket.gethostname()
    global_log_file = os.path.join(results_folder,
                                   get_log_time() + "-ucr-fcnn.log")
    with open(global_log_file, "a") as file:
        # Write the metadata.
        file.write("UCR datasets,final results,hostname," + str(
            hostname) + ",timestamp," + get_log_time() + ",num_epochs," + str(
            num_epochs) + "\n")
        # Write the header.
        file.write(
            "dataset,train_loss,train_accuracy,test_loss,test_accuracy\n")

    for ucr_dataset in flist:
        main(dataset_name=ucr_dataset)
