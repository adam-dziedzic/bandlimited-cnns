#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
@author: ady@uchicago.edu
"""

import os
import sys

import argparse
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import socket
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateauKeras
from keras.models import Model
from keras.utils import np_utils
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch
from torch.utils.data import Dataset
from torchvision import transforms

from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfft
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftAutograd
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftCompressSignalOnly
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftSimple
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftSimpleForLoop
from cnns.nnlib.pytorch_layers.AdamFloat16 import AdamFloat16
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import NextPower2
from cnns.nnlib.utils.general_utils import DynamicLossScale
from cnns.nnlib.utils.general_utils import DebugMode
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import mem_log_file
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel

# from memory_profiler import profile

try:
    import apex
except ImportError:
    raise ImportError("""Please install apex from 
    https://www.github.com/nvidia/apex to run this code.""")

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex.fp16_utils import network_to_half

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)

data_folder = "TimeSeriesDatasets"
# ucr_path = os.path.join(dir_path, os.pardir, data_folder)
ucr_path = os.path.join(os.pardir, data_folder)
results_folder = "results"

num_epochs = 300  # 300

plt.switch_backend('agg')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TimeSeries')
min_batch_size = 16
parser.add_argument('--min_batch_size', type=int, default=min_batch_size,
                    metavar='N',
                    help='input batch size for training (default: {})'.format(
                        min_batch_size))
parser.add_argument('--test_batch_size', type=int, default=min_batch_size,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=num_epochs, metavar='N',
                    help='number of epochs to train (default: {})'.format(
                        num_epochs))
learning_rate = 0.001
parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                    help='learning rate (default: {})'.format(
                        learning_rate))
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=31, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training '
                         'status')
parser.add_argument("--optimizer_type", default="ADAM",
                    # ADAM_FLOAT16, ADAM, MOMENTUM
                    help="the type of the optimizer, please choose from: " +
                         ",".join(OptimizerType.get_names()))
parser.add_argument("--memory_type", default="STANDARD",
                    # "STANDARD", "PINNED"
                    help="the type of the memory used, please choose from: " +
                         ",".join(MemoryType.get_names()))
parser.add_argument("-w", "--workers", default=4, type=int,
                    help="number of workers to fetch data for pytorch data "
                         "loader, 0 means that the data will be "
                         "loaded in the main process")
parser.add_argument("-n", "--net", default="fcnn",
                    help="the type of net: alexnet, densenet, resnet, fcnn.")
parser.add_argument("-d", "--datasets", default="all",
                    help="the type of datasets: all or debug.")
parser.add_argument("-i", "--index_back", default=0, type=int,
                    help="How many indexes (values) from the back of the "
                         "frequency representation should be discarded? This "
                         "is the compression in the FFT domain.")
parser.add_argument("-p", "--preserve_energy", default=99, type=float,
                    help="How much energy should be preserved in the "
                         "frequency representation of the signal? This "
                         "is the compression in the FFT domain.")
parser.add_argument("-b", "--mem_test", default=False, type=bool,
                    help="is it the memory test")
parser.add_argument("-a", "--is_data_augmentation", default=True, type=bool,
                    help="should the data augmentation be applied")
parser.add_argument("-g", "--is_debug", default="FALSE",  # TRUE or FALSE
                    help="is it the debug mode execution: " + ",".join(
                        DebugMode.get_names()))
parser.add_argument("--sample_count_limit", default=0, type=int,
                    help="number of samples taken from the dataset (0 - inactive)")
parser.add_argument("-c", "--conv_type", default="FFT1D",
                    # "FFT1D", "STANDARD". "AUTOGRAD", "SIMPLE_FFT"
                    help="the type of convolution, SPECTRAL_PARAM is with the "
                         "convolutional weights initialized in the spectral "
                         "domain, please choose from: " + ",".join(
                        ConvType.get_names()))
parser.add_argument("--compress_type", default="STANDARD",
                    # "STANDARD", "BIG_COEFF", "LOW_COEFF"
                    help="the type of compression to be applied: " + ",".join(
                        CompressType.get_names()))
parser.add_argument("--network_type", default="STANDARD",
                    # "STANDARD", "SMALL"
                    help="the type of network: " + ",".join(
                        NetworkType.get_names()))
parser.add_argument("--tensor_type", default="FLOAT32",
                    # "FLOAT32", "FLOAT16", "DOUBLE", "INT"
                    help="the tensor data type: " + ",".join(
                        TensorType.get_names()))
parser.add_argument("--next_power2", default="TRUE",
                    # "TRUE", "FALSE"
                    help="should we extend the input to the length of a power "
                         "of 2 before taking its fft? " + ",".join(
                        NextPower2.get_names()))
parser.add_argument('--static_loss_scale', type=float, default=1,
                    help="""Static loss scale, positive power of 2 values can 
                    improve fp16 convergence.""")
parser.add_argument('--dynamic_loss_scale', default="TRUE",
                    help='(bool) Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                         '--static-loss-scale. Options: ' + ",".join(
                        DynamicLossScale.get_names()))

args = parser.parse_args()

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

use_cuda = True if args.no_cuda is False else False
if torch.cuda.is_available() and use_cuda:
    print("conv1D_cuda is available: ")
    device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")

CONV_TYPE_ERROR = "Unknown type of convolution."


class FlatTransformation(object):
    """Transform a tensor to its flat representation

    Given tensor (C,H,W), will flatten it to (C, W) - a single data dimension.

    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Flattened tensor (C, W)
        """
        C, H, W = tensor.size()
        tensor = tensor.view(C, H * W)
        return tensor


class DtypeTransformation(object):
    """Transform a tensor to its dtype representation.

    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, tensor, dtype):
        """
        Args:
            tensor (Tensor): Tensor image.

        Returns:
            Tensor: with dtype.
        """
        return tensor.to(dtype=self.dtype)


def get_transform_train(dtype=torch.float32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        FlatTransformation(),
        DtypeTransformation(dtype)
    ])
    return transform_train


def get_transform_test(dtype=torch.float32):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        FlatTransformation(),
        DtypeTransformation(dtype),
    ])
    return transform_test


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


class Conv(object):

    def __init__(self, kernel_sizes, in_channels, out_channels, strides,
                 conv_pads, is_debug=True, dtype=None):
        """
        Create the convolution object from which we fetch the convolution
        operations.

        :param kernel_sizes: the sizes of the kernels in each conv layer.
        :param in_channels: number of channels in the input data.
        :param out_channels: the number of filters for each conv layer.
        :param strides: the strides for the convolutions.
        :param conv_pads: padding for each convolutional layer.
        :param is_debug: is the debug mode execution?
        """
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.conv_pads = conv_pads
        self.conv_type = ConvType[args.conv_type]
        self.index_back = args.index_back
        self.preserve_energy = args.preserve_energy
        self.is_debug = is_debug
        self.compress_type = CompressType[args.compress_type]

        self.dtype = dtype
        if self.dtype is None:
            tensor_type = TensorType[args.tensor_type]
            if tensor_type is TensorType.FLOAT32:
                self.dtype = torch.float32
            elif tensor_type is TensorType.FLOAT16:
                self.dtype = torch.float16
            elif tensor_type is TensorType.DOUBLE:
                self.dtype = torch.double
            else:
                raise ValueError(f"Unknown dtype: {tensor_type}")
            self.dtype = dtype

    def get_conv(self, index, index_back=None):
        if index == 0:
            in_channels = self.in_channels
        else:
            in_channels = self.out_channels[index - 1]

        if index_back is None:
            index_back = self.index_back

        if self.conv_type is ConvType.STANDARD:
            return nn.Conv1d(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=(self.conv_pads[index] // 2))
        elif self.conv_type is ConvType.FFT1D:
            next_power2 = NextPower2[args.next_power2]
            next_power2 = True if next_power2 is NextPower2.TRUE else False
            return Conv1dfft(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=(self.conv_pads[index] // 2),
                             index_back=index_back,
                             use_next_power2=next_power2,
                             conv_index=index,
                             preserve_energy=self.preserve_energy,
                             is_debug=self.is_debug,
                             compress_type=self.compress_type,
                             dtype=self.dtype)
        elif self.conv_type is ConvType.AUTOGRAD:
            return Conv1dfftAutograd(in_channels=in_channels,
                                     out_channels=self.out_channels[index],
                                     stride=self.strides[index],
                                     kernel_size=self.kernel_sizes[index],
                                     padding=(self.conv_pads[index] // 2),
                                     index_back=self.index_back)
        elif self.conv_type is ConvType.SIMPLE_FFT:
            return Conv1dfftSimple(in_channels=in_channels,
                                   out_channels=self.out_channels[index],
                                   stride=self.strides[index],
                                   kernel_size=self.kernel_sizes[index],
                                   padding=(self.conv_pads[index] // 2),
                                   index_back=self.index_back)
        elif self.conv_type is ConvType.SIMPLE_FFT_FOR_LOOP:
            return Conv1dfftSimpleForLoop(in_channels=in_channels,
                                          out_channels=self.out_channels[index],
                                          stride=self.strides[index],
                                          kernel_size=self.kernel_sizes[index],
                                          padding=(self.conv_pads[index] // 2),
                                          index_back=self.index_back)
        elif self.conv_type is ConvType.COMPRESS_INPUT_ONLY:
            return Conv1dfftCompressSignalOnly(
                in_channels=in_channels, out_channels=self.out_channels[index],
                stride=self.strides[index],
                kernel_size=self.kernel_sizes[index],
                padding=(self.conv_pads[index] // 2),
                index_back=self.index_back,
                preserve_energy=self.preserve_energy)
        else:
            raise CONV_TYPE_ERROR


class FCNNPytorch(nn.Module):

    # @profile
    def __init__(self, input_size, num_classes, in_channels,
                 dtype=torch.float32, kernel_sizes=[8, 5, 3],
                 out_channels=[128, 256, 128], strides=[1, 1, 1]):
        """
        Create the FCNN model in PyTorch.

        :param input_size: the length (width) of the time series.
        :param num_classes: number of output classes.
        :param in_channels: number of channels in the input data.
        :param dtype: global - the type of pytorch data/weights.
        :param kernel_sizes: the sizes of the kernels in each conv layer.
        :param out_channels: the number of filters for each conv layer.
        :param strides: the strides for the convolutions.
        """
        super(FCNNPytorch, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dtype = dtype
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.strides = strides
        self.conv_type = ConvType[args.conv_type]
        is_debug = True if DebugMode[args.is_debug] is DebugMode.TRUE else False
        self.is_debug = is_debug

        self.relu = nn.ReLU(inplace=True)
        # For the "same" mode for the convolution, pad the input.
        conv_pads = [kernel_size - 1 for kernel_size in kernel_sizes]

        conv = Conv(kernel_sizes=kernel_sizes, in_channels=in_channels,
                    out_channels=out_channels, strides=strides,
                    conv_pads=conv_pads, is_debug=self.is_debug)

        index = 0
        self.conv0 = conv.get_conv(index=index)
        self.bn0 = nn.BatchNorm1d(num_features=out_channels[index])

        index = 1
        self.conv1 = conv.get_conv(index=index)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels[index])

        index = 2
        self.conv2 = conv.get_conv(index=index)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels[index])
        self.lin = nn.Linear(out_channels[index], num_classes)

    def pad_out(self, out, index):
        """
        Pad the output to keep the size of the processed input the same through
        all the layers.

        :param out: the output of the previous layer
        :param index: index of the conv layer.
        :return:
        """
        if self.kernel_sizes[index] % 2 == 0:
            # If kernel size is even, add one more padding value on the right.
            out = F.pad(out, (0, 1), "constant", 0)
        return out

    def forward(self, out):
        """
        The forward pass through the network.

        :param out: the input data for the network.
        :return: the output class.
        """

        # 0th layer.
        index = 0
        out = self.pad_out(out, index)
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu(out)

        # 1st layer.
        index = 1
        out = self.pad_out(out, index)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 2nd layer.
        index = 2
        out = self.pad_out(out, index)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Classification.
        # Average across the channels.
        # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
        # In Keras it is implemented as: K.mean(inputs, axis=1). The channel is
        # the last dimension in Keras.
        out = torch.mean(out, dim=2)
        out = self.lin(out)

        # To imitate the cross entropy loss with the nll (negative log
        # likelihood) loss.
        out = log_softmax(out, dim=-1)

        return out


def getModelPyTorch(input_size, num_classes, in_channels, dtype=torch.float32):
    """
    Get the PyTorch version of the FCNN model.

    :param input_size: the length (width) of the time series.
    :param num_classes: number of output classes.
    :param in_channels: number of channels in the input data.
    :param dtype: global - the type of torch data/weights.

    :return: the model.
    """
    out_channels = None
    network_type = NetworkType[args.network_type]
    if network_type == NetworkType.SMALL:
        out_channels = [1, 1, 1]
    elif network_type == NetworkType.STANDARD:
        out_channels = [128, 256, 128]
    return FCNNPytorch(input_size=input_size, num_classes=num_classes,
                       in_channels=in_channels, out_channels=out_channels,
                       dtype=dtype)


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


def getData(fname):
    x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fname + '/' + fname + '_TEST')
    num_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] // 10, args.min_batch_size)

    y_train = ((y_train - y_train.min()) / (y_train.max() - y_train.min()) * (
            num_classes - 1)).astype(int)
    y_test = ((y_test - y_test.min()) / (y_test.max() - y_test.min()) * (
            num_classes - 1)).astype(int)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std

    x_test = (x_test - x_train_mean) / x_train_std
    # Add a single channels at the end of the data.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test, batch_size, num_classes


def run_keras():
    for each in flist:
        fname = each

        x_train, y_train, x_test, y_test, batch_size, num_classes = getData(
            fname=fname)

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
                         nb_epoch=args.epochs,
                         verbose=1, validation_data=(x_test, Y_test),
                         callbacks=[reduce_lr])

        # Print the testing results which has the lowest training loss.
        # Print the testing results which has the lowest training loss.
        log = pd.DataFrame(hist.history)
        print(log.loc[log['loss'].idxmin]['loss'],
              log.loc[log['loss'].idxmin]['val_acc'])


# @profile
def train(model, device, train_loader, optimizer, epoch,
          dtype=torch.float):
    """
    Train the model.

    :param model: deep learning model.
    :param device: cpu or gpu.
    :param train_loader: the training dataset.
    :param optimizer: Adam, Momemntum, etc.
    :param epoch: the current epoch number.
    :param dtype: the type of the tensors.
    """

    if dtype is torch.float16:
        """
        amp_handle: tells it where backpropagation occurs so that it can 
        properly scale the loss and clear internal per-iteration state.
        """
        # amp_handle = amp.init()
        # optimizer = amp_handle.wrap_optimizer(optimizer)
        dynamic_loss_scale = DynamicLossScale[args.dynamic_loss_scale]
        dynamic_loss_scale = True if dynamic_loss_scale is DynamicLossScale.TRUE else False

        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=True)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device, dtype=dtype), target.to(
            device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # The cross entropy loss combines `log_softmax` and `nll_loss` in
        # a single function.
        # loss = F.cross_entropy(output, target)

        if dtype is torch.float16:
            """
            https://github.com/NVIDIA/apex/tree/master/apex/amp
            
            Not used: at each optimization step in the training loop, perform 
            the following:
            Cast gradients to FP32. If a loss was scaled, descale the gradients.
            Apply updates in FP32 precision and copy the updated parameters to 
            the model, casting them to FP16.
            """

            # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            # loss = optimizer.scale_loss(loss)
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100.0 * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, dataset_type="test", dtype=torch.float):
    """

    :param model: deep learning model.
    :param device: cpu or gpu.
    :param test_loader: the input data.
    :param dataset_type: test or train.
    :param dtype: the data type of the tensor.
    :return: test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    counter = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device, dtype=dtype), target.to(
                device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= counter
        accuracy = 100. * correct / counter
        print(
            f"""{dataset_type} set: Average loss: {test_loss}, Accuracy: {correct}/{counter} ({accuracy}%)""")
        return test_loss, accuracy


# @profile
def main(dataset_name):
    """
    The main training.

    :param dataset_name: the name of the dataset from UCR.
    """
    is_debug = True if DebugMode[args.is_debug] is True else False
    dataset_log_file = os.path.join(
        results_folder, get_log_time() + "-" + dataset_name + "-fcnn.log")
    DATASET_HEADER = HEADER + ",dataset," + str(dataset_name) + "\n"
    with open(dataset_log_file, "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)
        # Write the header with the names of the columns.
        file.write(
            "epoch,train_loss,train_accuracy,test_loss,test_accuracy,epoch_time\n")

    with open(additional_log_file, "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)

    with open(mem_log_file, "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer_type = OptimizerType[args.optimizer_type]

    num_workers = args.workers
    pin_memory = False
    if MemoryType[args.memory_type] is MemoryType.PINNED:
        pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        if TensorType[args.tensor_type] is TensorType.FLOAT32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif TensorType[args.tensor_type] is TensorType.FLOAT16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        elif TensorType[args.tensor_type] is TensorType.DOUBLE:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        kwargs = {'num_workers': num_workers}
        if TensorType[args.tensor_type] is TensorType.FLOAT32:
            torch.set_default_tensor_type(torch.FloatTensor)
        elif TensorType[args.tensor_type] is TensorType.FLOAT16:
            torch.set_default_tensor_type(torch.HalfTensor)
        elif TensorType[args.tensor_type] is TensorType.DOUBLE:
            torch.set_default_tensor_type(torch.DoubleTensor)
    dtype = torch.float
    if TensorType[args.tensor_type] is TensorType.FLOAT16:
        dtype = torch.float16
    elif TensorType[args.tensor_type] is TensorType.DOUBLE:
        dtype = torch.double

    in_channels = None  # number of channels in the input data
    sample_count = args.sample_count_limit
    if dataset_name is "cifar10":
        num_classes = 10
        width = 32 * 32
        # standard batch size is 32
        batch_size = args.min_batch_size
        in_channels = 3  # number of channels in the input data
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True,
                                                     transform=get_transform_train(dtype=torch.float))
        if is_debug and sample_count > 0:
            train_dataset.train_data = train_dataset.train_data[:sample_count]
            train_dataset.train_labels = train_dataset.train_labels[
                                         :sample_count]

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   **kwargs)

        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True,
                                                    transform=get_transform_test(dtype=torch.float))
        if is_debug and sample_count > 0:
            test_dataset.test_data = test_dataset.test_data[:sample_count]
            test_dataset.test_labels = test_dataset.test_labels[:sample_count]
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  **kwargs)

    elif dataset_name in os.listdir(ucr_path):  # dataset from UCR archive
        in_channels = 1  # number of channels in the input data
        train_dataset = UCRDataset(dataset_name, train=True,
                                   transformations=transforms.Compose(
                                       [ToTensor(dtype=torch.float),
                                        AddChannel()]))
        if is_debug and sample_count > 0:
            train_dataset.set_length(sample_count)
        train_size = len(train_dataset)
        batch_size = args.min_batch_size
        if train_size < batch_size:
            batch_size = train_size
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            **kwargs)

        test_dataset = UCRDataset(dataset_name, train=False,
                                  transformations=transforms.Compose(
                                      [ToTensor(dtype=torch.float),
                                       AddChannel()]))
        if is_debug and sample_count > 0:
            test_dataset.set_length(sample_count)
        num_classes = test_dataset.num_classes
        width = test_dataset.width
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = getModelPyTorch(input_size=width, num_classes=num_classes,
                            in_channels=in_channels, dtype=dtype)
    model.to(device)
    if dtype is torch.float16:
        # model.half()  # convert to half precision
        model = network_to_half(model)
        """
        You want to make sure that the BatchNormalization layers use float32 for 
        accumulation or you will have convergence issues.
        https://discuss.pytorch.org/t/training-with-half-precision/11815
        """
        # for layer in model.modules():
        #     if isinstance(layer, nn.BatchNorm1d) or isinstance(layer,
        #                                                        nn.BatchNorm2d):
        #         layer.float()

    params = model.parameters()
    eps = 1e-8

    if optimizer_type is OptimizerType.MOMENTUM:
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif optimizer_type is OptimizerType.ADAM_FLOAT16:
        optimizer = AdamFloat16(params, lr=args.lr, eps=eps)
    else:
        optimizer = optim.Adam(params, lr=args.lr, eps=eps)



    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = ReduceLROnPlateauPyTorch(optimizer=optimizer, mode='min',
                                         factor=0.5, patience=50)

    train_loss = train_accuracy = test_loss = test_accuracy = 0.0
    # max = choose the best model.
    min_train_loss = min_test_loss = sys.float_info.max
    max_train_accuracy = max_test_accuracy = 0.0
    dataset_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print("training for epoch: ", epoch)
        train(model=model, device=device, train_loader=train_loader,
              optimizer=optimizer, epoch=epoch, dtype=dtype)
        print("test train set for epoch: ", epoch)
        train_loss, train_accuracy = test(model=model, device=device,
                                          test_loader=train_loader,
                                          dataset_type="train", dtype=dtype)
        print("test test set for epoch: ", epoch)
        test_loss, test_accuracy = test(model=model, device=device,
                                        test_loader=test_loader,
                                        dataset_type="test", dtype=dtype)
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        scheduler.step(train_loss)

        with open(dataset_log_file, "a") as file:
            file.write(str(epoch) + "," + str(train_loss) + "," + str(
                train_accuracy) + "," + str(test_loss) + "," + str(
                test_accuracy) + "," + str(
                time.time() - epoch_start_time) + "\n")

        # Metric: select the best model based on the best train loss (minimal).
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            max_train_accuracy = train_accuracy
            min_test_loss = test_loss
            max_test_accuracy = test_accuracy

    with open(global_log_file, "a") as file:
        file.write(dataset_name + "," + str(min_train_loss) + "," + str(
            max_train_accuracy) + "," + str(min_test_loss) + "," + str(
            max_test_accuracy) + "," + str(
            time.time() - dataset_start_time) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    hostname = socket.gethostname()
    global_log_file = os.path.join(results_folder,
                                   get_log_time() + "-ucr-fcnn.log")
    args_dict = vars(args)
    args_str = ",".join(
        [str(key) + ":" + str(value) for key, value in args_dict.items()])
    HEADER = "UCR datasets,final results,hostname," + str(
        hostname) + ",timestamp," + get_log_time() + "," + str(args_str)
    with open(additional_log_file, "a") as file:
        # Write the metadata.
        file.write(HEADER + "\n")
    with open(global_log_file, "a") as file:
        # Write the metadata.
        file.write(HEADER)
        # Write the header.
        file.write(
            "dataset,min_train_loss,max_train_accuracy,min_test_loss,max_test_accuracy,execution_time\n")

    if args.datasets == "all":
        flist = os.listdir(ucr_path)
    elif args.datasets == "debug":
        # flist = ["50words"]
        # flist = ["cifar10"]
        # flist = ["zTest"]
        # flist = ["zTest50words"]
        # flist = ["InlineSkate"]
        # flist = ["Adiac"]
        # flist = ["HandOutlines"]
        flist = ["ztest"]
        # flist = ["Cricket_X"]
        # flist = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly',
        #         'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
        #         'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X',
        #         'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
        #         'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        #         'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000',
        #         'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
        #         'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham',
        #         'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
        #         'InsectWingbeatSound', 'ItalyPowerDemand',
        #         'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT',
        #         'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
        #         'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
        #         'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
        #         'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
        #         'Plane', 'ProximalPhalanxOutlineAgeGroup',
        #         'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        #         'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
        #         'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
        #         'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
        #         'SwedishLeaf', 'Symbols', 'synthetic_control',
        #         'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'Two_Patterns',
        #         'TwoLeadECG', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
        #         'uWaveGestureLibrary_Z', 'UWaveGestureLibraryAll', 'wafer',
        #         'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga',
        #         'ztest']
        # flist = ['DiatomSizeReduction',
        #          'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        #          'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000',
        #          'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
        #          'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham',
        #          'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
        #          'InsectWingbeatSound', 'ItalyPowerDemand',
        #          'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT',
        #          'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
        #          'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
        #          'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
        #          'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
        #          'Plane', 'ProximalPhalanxOutlineAgeGroup',
        #          'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        #          'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
        #          'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
        #          'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
        #          'SwedishLeaf', 'Symbols', 'synthetic_control',
        #          'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
        #          'Two_Patterns',
        #          'TwoLeadECG', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
        #          'uWaveGestureLibrary_Z', 'UWaveGestureLibraryAll', 'wafer',
        #          'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga',
        #          'ztest']
    else:
        raise AttributeError("Unknown datasets: ", args.datasets)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flist = sorted(flist, key=lambda s: s.lower())
    flist = flist[3:]  # start from Beef
    # reversed(flist)
    print("flist: ", flist)
    for dataset_name in flist:
        print("Dataset: ", dataset_name)
        with open(additional_log_file, "a") as file:
            file.write(dataset_name + "\n")
        main(dataset_name=dataset_name)

    print("total elapsed time (sec): ", time.time() - start_time)
