#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
@author: ady@uchicago.edu
"""

import os
import sys
import pathlib
import logging

import argparse
import numpy as np
import socket
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch

from cnns.nnlib.pytorch_layers.AdamFloat16 import AdamFloat16
from cnns.nnlib.pytorch_architecture.le_net import LeNet
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.pytorch_architecture.fcnn import FCNNPytorch
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import NextPower2
from cnns.nnlib.utils.general_utils import Visualize
from cnns.nnlib.utils.general_utils import DynamicLossScale
from cnns.nnlib.utils.general_utils import DebugMode
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import mem_log_file
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.mnist import get_mnist
from cnns.nnlib.datasets.cifar10 import get_cifar10
from cnns.nnlib.datasets.ucr.ucr import get_ucr

# from memory_profiler import profile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]

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

results_folder_name = "results"
results_dir = os.path.join(os.curdir, results_folder_name)
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

models_folder_name = "models"
models_dir = os.path.join(os.curdir, models_folder_name)
pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

num_epochs = 3  # 300

# plt.switch_backend('agg')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TimeSeries')
min_batch_size = 64
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
                    help='random seed (default: 31)')
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
parser.add_argument("--model_path",
                    default="no_model",
                    # default = "2018-11-07-00-00-27-dataset-50words-preserve-energy-90-test-accuracy-58.46153846153846.model",
                    # default="2018-11-06-21-05-48-dataset-50words-preserve-energy-90-test-accuracy-12.5.model",
                    # default="2018-11-06-21-19-51-dataset-50words-preserve-energy-90-test-accuracy-12.5.model",
                    # no_model
                    # 2018-11-06-21-05-48-dataset-50words-preserve-energy-90-test-accuracy-12.5.model
                    help="The path to a saved model.")
parser.add_argument("-d", "--datasets", default="debug",
                    help="the type of datasets: all, debug, cifar10, mnist.")
parser.add_argument("-i", "--index_back", default=0, type=int,
                    help="How many indexes (values) from the back of the "
                         "frequency representation should be discarded? This "
                         "is the compression in the FFT domain.")
parser.add_argument('--preserve_energy', nargs="+", type=int,
                    default=[90],
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
parser.add_argument("--sample_count_limit", default=64, type=int,
                    help="number of samples taken from the dataset "
                         "(0 - inactive)")
parser.add_argument("-c", "--conv_type", default="FFT2D",
                    # "FFT1D", "FFT2D", "STANDARD", "STANDARD2D", "AUTOGRAD",
                    # "SIMPLE_FFT"
                    help="the type of convolution, SPECTRAL_PARAM is with the "
                         "convolutional weights initialized in the spectral "
                         "domain, please choose from: " + ",".join(
                        ConvType.get_names()))
parser.add_argument("--compress_type", default="STANDARD",
                    # "STANDARD", "BIG_COEFF", "LOW_COEFF"
                    help="the type of compression to be applied: " + ",".join(
                        CompressType.get_names()))
parser.add_argument("--network_type", default="ResNet18",
                    # "FCNN_STANDARD", "FCNN_SMALL", "LE_NET", "ResNet18"
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
parser.add_argument("--visualize", default="FALSE",
                    # "TRUE", "FALSE"
                    help="should we visualize the activations map after each "
                         "of the convolutional layers? " + ",".join(
                        Visualize.get_names()))
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


def getModelPyTorch(input_size, num_classes, in_channels, out_channels=None,
                    dtype=torch.float32, batch_size=args.min_batch_size,
                    preserve_energy=100, flat_size=320, is_debug=False,
                    args=args):
    """
    Get the PyTorch version of the FCNN model.

    :param input_size: the length (width) of the time series.
    :param num_classes: number of output classes.
    :param in_channels: number of channels in the input data for a convolution.
    :param out_channels: number of channels in the output of a convolution.
    :param dtype: global - the type of torch data/weights.
    :param flat_size: the size of the flat vector after the conv layers.
    :return: the model.
    """
    network_type = NetworkType[args.network_type]
    if network_type == NetworkType.LE_NET:
        return LeNet(input_size=input_size, args=args, num_classes=num_classes,
                     in_channels=in_channels, out_channels=out_channels,
                     dtype=dtype, batch_size=batch_size,
                     preserve_energy=preserve_energy, flat_size=flat_size,
                     is_debug=is_debug)
    elif network_type == NetworkType.FCNN_SMALL or (
            network_type == NetworkType.FCNN_STANDARD):
        if network_type == NetworkType.FCNN_SMALL:
            out_channels = [1, 1, 1]
        elif network_type == NetworkType.FCNN_STANDARD:
            out_channels = [128, 256, 128]
        return FCNNPytorch(input_size=input_size, num_classes=num_classes,
                           in_channels=in_channels, out_channels=out_channels,
                           dtype=dtype, preserve_energy=preserve_energy)
    elif network_type == NetworkType.ResNet18:
        return resnet18(num_classes=num_classes, args=args)
    else:
        raise Exception("Unknown network_type: ", network_type)


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


# @profile
def train(model, device, train_loader, optimizer, epoch, dtype=torch.float):
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
def main(dataset_name, preserve_energy):
    """
    The main training.

    :param dataset_name: the name of the dataset from UCR.
    """
    is_debug = True if DebugMode[args.is_debug] is DebugMode.TRUE else False
    dataset_log_file = os.path.join(
        results_folder_name, get_log_time() + "-dataset-" + str(dataset_name) + \
                             "-preserve-energy-" + str(preserve_energy) + \
                             ".log")
    DATASET_HEADER = HEADER + ",dataset," + str(dataset_name) + \
                     "-current-preserve-energy-" + str(preserve_energy) + "\n"
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

    batch_size = args.min_batch_size

    if dataset_name is "cifar10":
        train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels = get_cifar10(
            args)
    elif dataset_name is "mnist":
        train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels = get_mnist(
            args)
    elif dataset_name in os.listdir(ucr_path):  # dataset from UCR archive
        train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels = get_ucr(
            args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    args.in_channels = in_channels

    model = getModelPyTorch(input_size=width, num_classes=num_classes,
                            in_channels=in_channels, out_channels=out_channels,
                            dtype=dtype, batch_size=batch_size,
                            preserve_energy=preserve_energy,
                            flat_size=flat_size, is_debug=is_debug, args=args)

    # https://pytorch.org/docs/master/notes/serialization.html
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=device))
        msg = "loaded model: " + args.model_path
        logger.info(msg)
        print(msg)

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

    if Visualize[args.visualize] is True and is_debug is True:
        test_loss, test_accuracy = test(model, test_loader=test_loader,
                                        data_type="test", dtype=dtype)
        with open(global_log_file, "a") as file:
            file.write(
                "visualize," + dataset_name + "," + str(test_loss) + "," + str(
                    test_accuracy) + "," + str(get_log_time()) + "\n")
        return

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

            model_path = os.path.join(models_dir,
                                      get_log_time() + "-dataset-" + str(
                                          dataset_name) + \
                                      "-preserve-energy-" + str(
                                          preserve_energy) + \
                                      "-test-accuracy-" + str(
                                          test_accuracy) + ".model")
            torch.save(model.state_dict(), model_path)

    with open(global_log_file, "a") as file:
        file.write(dataset_name + "," + str(min_train_loss) + "," + str(
            max_train_accuracy) + "," + str(min_test_loss) + "," + str(
            max_test_accuracy) + "," + str(
            time.time() - dataset_start_time) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    hostname = socket.gethostname()
    global_log_file = os.path.join(results_folder_name,
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
    elif args.datasets == "cifar10":
        flist = ["cifar10"]
    elif args.datasets == "mnist":
        flist = ["mnist"]
    elif args.datasets == "debug":
        # flist = ["50words"]
        flist = ["cifar10"]
        # flist = ["mnist"]
        # flist = ["zTest"]
        # flist = ["zTest50words"]
        # flist = ["InlineSkate"]
        # flist = ["Adiac"]
        # flist = ["HandOutlines"]
        # flist = ["ztest"]
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
    # flist = flist[3:]  # start from Beef
    # reversed(flist)
    print("flist: ", flist)
    preserve_energies = args.preserve_energy
    for dataset_name in flist:
        for preserve_energy in preserve_energies:
            print("Dataset: ", dataset_name)
            print("preserve energy: ", preserve_energy)
            args.preserve_energy = preserve_energy
            with open(global_log_file, "a") as file:
                file.write("dataset name: " + dataset_name + "\n" +
                           "preserve energy: " + str(preserve_energy) + "\n")
            main(dataset_name=dataset_name, preserve_energy=preserve_energy)

            print("total elapsed time (sec): ", time.time() - start_time)
