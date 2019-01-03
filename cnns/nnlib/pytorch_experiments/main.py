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
import traceback

import argparse
import numpy as np
import socket
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch
from torch.optim.lr_scheduler import MultiStepLR

from cnns.nnlib.pytorch_layers.AdamFloat16 import AdamFloat16
from cnns.nnlib.pytorch_architecture.le_net import LeNet
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.pytorch_architecture.densenet import densenet_cifar
from cnns.nnlib.pytorch_architecture.fcnn import FCNNPytorch
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import SchedulerType
from cnns.nnlib.utils.general_utils import LossType
from cnns.nnlib.utils.general_utils import LossReduction
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import Bool
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import mem_log_file
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.mnist import get_mnist
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.ucr.ucr import get_ucr
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.pytorch_experiments.utils.progress_bar import progress_bar

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
# print("current working directory: ", dir_path)

ucr_data_folder = "TimeSeriesDatasets"
# ucr_path = os.path.join(dir_path, os.pardir, data_folder)
ucr_path = os.path.join(os.pardir, os.pardir, ucr_data_folder)

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

# plt.switch_backend('agg')

# Execution parameters.
args = Arguments()
parser = argparse.ArgumentParser(description='PyTorch TimeSeries')
parser.add_argument('--min_batch_size', type=int, default=args.min_batch_size,
                    help=f"input mini batch size for training "
                    f"(default: {args.min_batch_size})")
parser.add_argument('--test_batch_size', type=int, default=args.test_batch_size,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=args.epochs, metavar='Epochs',
                    help=f"number of epochs to train ("
                    f"default: {args.epochs})")
parser.add_argument('--learning_rate', type=float,
                    default=args.learning_rate, metavar='LR',
                    help=f'learning rate (default: {args.learning_rate})')
parser.add_argument('--weight_decay', type=float, default=args.weight_decay,
                    help=f'weight decay (default: {args.weight_decay})')
parser.add_argument('--momentum', type=float, default=args.momentum,
                    metavar='M',
                    help=f'SGD momentum (default: {args.momentum})')
parser.add_argument('--use_cuda', default="TRUE" if args.use_cuda else "FALSE",
                    help="use CUDA for training and inference; "
                         "options: " + ",".join(Bool.get_names()))
parser.add_argument('--seed', type=int, default=args.seed, metavar='S',
                    help='random seed (default: 31)')
parser.add_argument('--log-interval', type=int, default=args.log_interval,
                    metavar='N',
                    help='how many batches to wait before logging training '
                         'status')
parser.add_argument("--optimizer_type", default=args.optimizer_type.name,
                    # ADAM_FLOAT16, ADAM, MOMENTUM
                    help="the type of the optimizer, please choose from: " +
                         ",".join(OptimizerType.get_names()))
parser.add_argument("--scheduler_type", default=args.scheduler_type.name,
                    # StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR
                    # CosineAnnealingLR
                    help="the type of the scheduler, please choose from: " +
                         ",".join(SchedulerType.get_names()))
parser.add_argument("--loss_type", default=args.loss_type.name,
                    # StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR
                    # CosineAnnealingLR
                    help="the type of the loss, please choose from: " +
                         ",".join(LossType.get_names()))
parser.add_argument("--loss_reduction", default=args.loss_reduction.name,
                    help="the type of the loss, please choose from: " +
                         ",".join(LossReduction.get_names()))
parser.add_argument("--memory_type", default=args.memory_type.name,
                    # "STANDARD", "PINNED"
                    help="the type of the memory used, please choose from: " +
                         ",".join(MemoryType.get_names()))
parser.add_argument("--workers", default=args.workers, type=int,
                    help="number of workers to fetch data for pytorch data "
                         "loader, 0 means that the data will be "
                         "loaded in the main process")
parser.add_argument("--model_path",
                    default=args.model_path,
                    # default = "2018-11-07-00-00-27-dataset-50words-preserve-energy-90-test-accuracy-58.46153846153846.model",
                    # default="2018-11-06-21-05-48-dataset-50words-preserve-energy-90-test-accuracy-12.5.model",
                    # default="2018-11-06-21-19-51-dataset-50words-preserve-energy-90-test-accuracy-12.5.model",
                    # no_model
                    # 2018-11-06-21-05-48-dataset-50words-preserve-energy-90-test-accuracy-12.5.model
                    help="The path to a saved model.")
parser.add_argument("--dataset", default=args.dataset,
                    help="the type of datasets: all, debug, cifar10, mnist.")
parser.add_argument("--index_back", default=args.index_back, type=float,
                    help="Percentage of indexes (values) from the back of the "
                         "frequency representation that should be discarded. "
                         "This is the compression in the FFT domain.")
parser.add_argument('--preserve_energies', nargs="+", type=float,
                    default=args.preserve_energies,
                    help="How much energy should be preserved in the "
                         "frequency representation of the signal? This "
                         "is the compression in the FFT domain.")
parser.add_argument("--mem_test",
                    default="TRUE" if args.mem_test else "FALSE",
                    help="is it the memory test; options: " + ",".join(
                        Bool.get_names()))
parser.add_argument("--is_data_augmentation",
                    default="TRUE" if args.is_data_augmentation else "FALSE",
                    help="should the data augmentation be applied; "
                         "options: " + ",".join(Bool.get_names()))
parser.add_argument("--is_debug",
                    default="TRUE" if args.is_debug else "FALSE",
                    help="is it the debug mode execution: " + ",".join(
                        Bool.get_names()))
parser.add_argument("--sample_count_limit", default=args.sample_count_limit,
                    type=int,
                    help="number of samples taken from the dataset "
                         "(0 - inactive)")
parser.add_argument("--conv_type", default=args.conv_type.name,
                    # "FFT1D", "FFT2D", "STANDARD", "STANDARD2D", "AUTOGRAD",
                    # "SIMPLE_FFT"
                    help="the type of convolution, SPECTRAL_PARAM is with the "
                         "convolutional weights initialized in the spectral "
                         "domain, please choose from: " + ",".join(
                        ConvType.get_names()))
parser.add_argument("--conv_exec_type",
                    default=args.conv_exec_type.name,
                    help="the type of internal execution of the convolution"
                         "operation, for example: SERIAL: is each data point "
                         "convolved separately with all the filters, all a "
                         "batch of datapoint is convolved with all filters in "
                         "one go; CUDA: the tensor element wise complex "
                         "multiplication is done on CUDA, choose options from: "
                         "" + ",".join(ConvExecType.get_names()))
parser.add_argument("--compress_type", default=args.compress_type.name,
                    # "STANDARD", "BIG_COEFF", "LOW_COEFF"
                    help="the type of compression to be applied: " + ",".join(
                        CompressType.get_names()))
parser.add_argument("--network_type", default=args.network_type.name,
                    # "FCNN_STANDARD", "FCNN_SMALL", "LE_NET", "ResNet18"
                    help="the type of network: " + ",".join(
                        NetworkType.get_names()))
parser.add_argument("--tensor_type", default=args.tensor_type.name,
                    # "FLOAT32", "FLOAT16", "DOUBLE", "INT"
                    help="the tensor data type: " + ",".join(
                        TensorType.get_names()))
parser.add_argument("--next_power2",
                    default="TRUE" if args.next_power2 else "FALSE",
                    # "TRUE", "FALSE"
                    help="should we extend the input to the length of a power "
                         "of 2 before taking its fft? " + ",".join(
                        Bool.get_names()))
parser.add_argument("--visualize", default="TRUE" if args.visulize else "FALSE",
                    # "TRUE", "FALSE"
                    help="should we visualize the activations map after each "
                         "of the convolutional layers? " + ",".join(
                        Bool.get_names()))
parser.add_argument('--static_loss_scale', type=float,
                    default=args.static_loss_scale,
                    help="""Static loss scale, positive power of 2 values can 
                    improve fp16 convergence.""")
parser.add_argument('--dynamic_loss_scale',
                    default="TRUE" if args.dynamic_loss_scale else "FALSE",
                    help="(bool) Use dynamic loss scaling. "
                         "If supplied, this argument supersedes " +
                         "--static-loss-scale. Options: " + ",".join(
                        Bool.get_names()))
parser.add_argument('--memory_size', type=float,
                    default=args.memory_size,
                    help="""GPU or CPU memory size in GB.""")
parser.add_argument("--is_progress_bar",
                    default="TRUE" if args.is_progress_bar else "FALSE",
                    # "TRUE", "FALSE"
                    help="should we show the progress bar after each batch was"
                         "processed in training and testing? " + ",".join(
                        Bool.get_names()))
parser.add_argument("--stride_type", default=args.stride_type.name,
                    # "FLOAT32", "FLOAT16", "DOUBLE", "INT"
                    help="the tensor data type: " + ",".join(
                        StrideType.get_names()))
parser.add_argument("--is_dev_dataset",
                    default="TRUE" if args.is_dev_dataset else "FALSE",
                    help="is it the dev set extracted from the train set, "
                         "default is {args.is_dev_set}, but choose param from: "
                         "" + ",".join(Bool.get_names()))
parser.add_argument("--dev_percent", default=args.dev_percent,
                    type=int,
                    help="percent of train set used as the development set"
                         " (range from 0 to 100), 0 - it is inactive,"
                         " default: {args.dev_percent}")
parser.add_argument("--adam_beta1", default=args.adam_beta1,
                    type=float,
                    help="beta1 value for the ADAM optimizer, default: "
                         "{args.adam_beta1}")
parser.add_argument("--adam_beta2", default=args.adam_beta2,
                    type=float,
                    help="beta2 value for the ADAM optimizer, default: "
                         "{args.adam_beta1}")
parser.add_argument("--cuda_block_threads", default=args.cuda_block_threads,
                    type=int, help="Max number of threads for a cuda block.")

parsed_args = parser.parse_args()
args.set_parsed_args(parsed_args=parsed_args)

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

if torch.cuda.is_available() and args.use_cuda:
    print("cuda is available: ")
    device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")


def getModelPyTorch(args=args):
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
    network_type = args.network_type
    if network_type is NetworkType.LE_NET:
        return LeNet(args=args)
    elif network_type is NetworkType.FCNN_SMALL or (
            network_type is NetworkType.FCNN_STANDARD):
        if network_type is NetworkType.FCNN_SMALL:
            args.out_channels = [1, 1, 1]
        elif network_type is NetworkType.FCNN_STANDARD:
            args.out_channels = [128, 256, 128]
        return FCNNPytorch(args=args)
    elif network_type == NetworkType.ResNet18:
        return resnet18(args=args)
    elif network_type == NetworkType.DenseNetCifar:
        return densenet_cifar(args=args)
    else:
        raise Exception("Unknown network_type: ", network_type)


def readucr(filename, data_type):
    parent_path = os.path.split(os.path.abspath(dir_path))[0]
    print("parent path: ", parent_path)
    filepath = os.path.join(parent_path, ucr_data_folder, filename,
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
def train(model, device, train_loader, optimizer, loss_function, epoch, args):
    """
    Train the model.

    :param model: deep learning model.
    :param device: cpu or gpu.
    :param train_loader: the training dataset.
    :param optimizer: Adam, Momemntum, etc.
    :param epoch: the current epoch number.
    :param
    """

    if args.dtype is torch.float16:
        """
        amp_handle: tells it where backpropagation occurs so that it can 
        properly scale the loss and clear internal per-iteration state.
        """
        # amp_handle = amp.init()
        # optimizer = amp_handle.wrap_optimizer(optimizer)

        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   verbose=True)

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device, dtype=args.dtype), target.to(
            device=device)
        optimizer.zero_grad()
        output = model(data)
        with open(additional_log_file, "a") as file:
            file.write("\n")
        loss = loss_function(output, target)

        # The cross entropy loss combines `log_softmax` and `nll_loss` in
        # a single function.
        # loss = F.cross_entropy(output, target)

        if args.dtype is torch.float16:
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

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100.0 * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if args.is_progress_bar:
            progress_bar(total, len(train_loader.dataset), epoch=epoch,
                         msg="Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)" %
                             (train_loss / total, 100. * correct / total,
                              correct,
                              total))

    # Test loss for the whole dataset.
    train_loss /= total
    accuracy = 100. * correct / total

    return train_loss, accuracy


def test(model, device, test_loader, loss_function, args, epoch=None):
    """
    Test the model and return test loss and accuracy.

    :param model: deep learning model.
    :param device: cpu or gpu.
    :param test_loader: the input data.
    :param dataset_type: test or train.
    :param dtype: the data type of the tensor.
    :param epoch: current epoch of the model training/testing.
    :return: test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device=device, dtype=args.dtype), target.to(
                device)
            output = model(data)
            with open(additional_log_file, "a") as file:
                file.write("\n")
            test_loss += loss_function(output,
                                       target).item()  # sum up batch loss
            # get the index of the max log-probability
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if args.is_progress_bar:
                progress_bar(total, len(test_loader.dataset), epoch=epoch,
                             msg="Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)" %
                                 (test_loss / total, 100. * correct / total,
                                  correct,
                                  total))

        # Test loss for the whole dataset.
        test_loss /= total
        accuracy = 100. * correct / total
        return test_loss, accuracy

# @profile
def main(args):
    """
    The main training.

    :param dataset_name: the name of the dataset from UCR.
    """
    is_debug = args.is_debug
    dataset_name = args.dataset_name
    preserve_energy = args.preserve_energy

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
            "epoch,train_loss,train_accuracy,dev_loss,dev_accuracy,test_loss,"
            "test_accuracy,epoch_time\n")

    with open(os.path.join(results_dir, additional_log_file), "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)

    with open(os.path.join(results_dir, mem_log_file), "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)

    torch.manual_seed(args.seed)
    optimizer_type = args.optimizer_type
    scheduler_type = args.scheduler_type
    loss_type = args.loss_type
    loss_reduction = args.loss_reduction

    use_cuda = args.use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    tensor_type = args.tensor_type
    if use_cuda:
        if tensor_type is TensorType.FLOAT32:
            cuda_type = torch.cuda.FloatTensor
        elif tensor_type is TensorType.FLOAT16:
            cuda_type = torch.cuda.HalfTensor
        elif tensor_type is TensorType.DOUBLE:
            cuda_type = torch.cuda.DoubleTensor
        else:
            raise Exception(f"Unknown tensor type: {tensor_type}")
        torch.set_default_tensor_type(cuda_type)
    else:
        if tensor_type is TensorType.FLOAT32:
            cpu_type = torch.FloatTensor
        elif tensor_type is TensorType.FLOAT16:
            cpu_type = torch.HalfTensor
        elif tensor_type is TensorType.DOUBLE:
            cpu_type = torch.DoubleTensor
        else:
            raise Exception(f"Unknown tensor type: {tensor_type}")
        torch.set_default_tensor_type(cpu_type)

    if tensor_type is TensorType.FLOAT32:
        dtype = torch.float32
    elif tensor_type is TensorType.FLOAT16:
        dtype = torch.float16
    elif tensor_type is TensorType.DOUBLE:
        dtype = torch.double
    else:
        raise Exception(f"Unknown tensor type: {tensor_type}")
    args.dtype = dtype

    train_loader, dev_loader, test_loader = None, None, None
    if dataset_name is "cifar10" or dataset_name is "cifar100":
        train_loader, test_loader = get_cifar(args, dataset_name)
    elif dataset_name is "mnist":
        train_loader, test_loader = get_mnist(args)
    elif dataset_name in os.listdir(ucr_path):  # dataset from UCR archive
        train_loader, test_loader, dev_loader = get_ucr(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = getModelPyTorch(args=args)
    model.to(device)
    model = torch.nn.DataParallel(model)

    # https://pytorch.org/docs/master/notes/serialization.html
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=device))
        msg = "loaded model: " + args.model_path
        # logger.info(msg)
        print(msg)
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
        optimizer = optim.SGD(params, lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif optimizer_type is OptimizerType.ADAM_FLOAT16:
        optimizer = AdamFloat16(params, lr=args.learning_rate, eps=eps)
    elif optimizer_type is OptimizerType.ADAM:
        optimizer = optim.Adam(params, lr=args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=args.weight_decay, eps=eps)
    else:
        raise Exception(f"Unknown optimizer type: {optimizer_type.name}")

    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type is SchedulerType.ReduceLROnPlateau:
        scheduler = ReduceLROnPlateauPyTorch(optimizer=optimizer, mode='min',
                                             factor=0.1, patience=10)
    elif scheduler_type is SchedulerType.MultiStepLR:
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[150, 250])
    else:
        raise Exception(f"Unknown scheduler type: {scheduler_type}")

    if loss_reduction is LossReduction.ELEMENTWISE_MEAN:
        reduction_function = "mean"
    elif loss_reduction is LossReduction.SUM:
        reduction_function = "sum"
    else:
        raise Exception(f"Unknown loss reduction: {loss_reduction}")

    if loss_type is LossType.CROSS_ENTROPY:
        loss_function = torch.nn.CrossEntropyLoss(reduction=reduction_function)
    elif loss_type is LossType.NLL:
        loss_function = torch.nn.NLLLoss(reduction=reduction_function)
    else:
        raise Exception(f"Unknown loss type: {loss_type}")

    train_loss = train_accuracy = test_loss = test_accuracy = 0.0
    # max = choose the best model.
    min_train_loss = min_test_loss = min_dev_loss = sys.float_info.max
    max_train_accuracy = max_test_accuracy = max_dev_accuracy = 0.0

    if args.visulize is True:
        start_visualize_time = time.time()
        test_loss, test_accuracy = test(
            model=model, device=device, test_loader=test_loader,
            loss_function=loss_function, args=args)
        elapsed_time = time.time() - start_visualize_time
        with open(global_log_file, "a") as file:
            file.write(
                dataset_name + ",None,None,None,None," + str(
                    test_loss) + "," + str(test_accuracy) + "," + str(
                    elapsed_time) + ",visualize" + "\n")
        return

    dataset_start_time = time.time()
    dev_loss = dev_accuracy = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print("\ntrain:")
        train_loss, train_accuracy = train(
            model=model, device=device, train_loader=train_loader, args=args,
            optimizer=optimizer, loss_function=loss_function, epoch=epoch)
        if args.is_dev_dataset:
            if dev_loader is None:
                raise Exception("The dev_loader was not set! Check methods to"
                                "get the data, e.g. get_ucr()")
            dev_loss, dev_accuracy = test(
                model=model, device=device, test_loader=dev_loader,
                loss_function=loss_function, args=args)
        print("\ntest:")
        test_loss, test_accuracy = test(
            model=model, device=device, test_loader=test_loader,
            loss_function=loss_function, args=args)
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        scheduler.step(train_loss)

        with open(dataset_log_file, "a") as file:
            file.write(str(epoch) + "," + str(train_loss) + "," + str(
                train_accuracy) + "," + str(dev_loss) + "," + str(
                dev_accuracy) + "," + str(test_loss) + "," + str(
                test_accuracy) + "," + str(
                time.time() - epoch_start_time) + "\n")

        # Metric: select the best model based on the best train loss (minimal).
        if args.is_dev_dataset:
            if dev_accuracy > max_dev_accuracy:
                min_train_loss = train_loss
                max_train_accuracy = train_accuracy
                min_dev_loss = dev_loss
                max_dev_accuracy = dev_accuracy
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
        else:
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                max_train_accuracy = train_accuracy
                min_dev_loss = dev_loss
                max_dev_accuracy = dev_accuracy
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
            max_train_accuracy) + "," + str(min_dev_loss) + "," + str(
            max_dev_accuracy) + "," + str(min_test_loss) + "," + str(
            max_test_accuracy) + "," + str(
            time.time() - dataset_start_time) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    hostname = socket.gethostname()
    global_log_file = os.path.join(results_folder_name,
                                   get_log_time() + "-ucr-fcnn.log")
    args_str = args.get_str()
    HEADER = "hostname," + str(
        hostname) + ",timestamp," + get_log_time() + "," + str(args_str)
    with open(additional_log_file, "a") as file:
        # Write the metadata.
        file.write(HEADER + "\n")
    with open(global_log_file, "a") as file:
        # Write the metadata.
        file.write(HEADER + "\n")
        file.write(
            "preserve energies: " +
            ",".join(
                [str(energy) for energy in args.preserve_energies]) +
            "\n")
        file.write(
            "dataset,"
            "min_train_loss,max_train_accuracy,"
            "min_dev_loss,max_dev_accuracy,"
            "min_test_loss,max_test_accuracy,"
            "execution_time\n")

    if args.dataset == "all" or args.dataset == "ucr":
        flist = os.listdir(ucr_path)
    elif args.dataset == "cifar10":
        flist = ["cifar10"]
    elif args.dataset == "cifar100":
        flist = ["cifar100"]
    elif args.dataset == "mnist":
        flist = ["mnist"]
    elif args.dataset == "debug":
        flist = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly',
                 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
                 'CinC_ECG_torso', 'Coffee', 'Computers']
        # flist = ["WIFI"]
        # flist = ["50words"]
        # flist = ["yoga"]
        # flist = ["Two_Patterns"]
        # flist = ["uWaveGestureLibrary_Z"]
        # flist = ["cifar10"]
        # flist = ["mnist"]
        # flist = ["zTest"]
        # flist = ["zTest50words"]
        # flist = ["InlineSkate"]
        # flist = ["Adiac"]
        # flist = ["HandOutlines"]
        # flist = ["ztest"]
        # flist = ["Cricket_X"]
        # flist = ["50words"]
        # flist = ["SwedishLeaf"]
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
    elif args.dataset == "debug0":
        flist = ['Cricket_X',
                 'Cricket_Y',
                 'Cricket_Z',
                 'DiatomSizeReduction',
                 'DistalPhalanxOutlineAgeGroup',
                 'DistalPhalanxOutlineCorrect',
                 'DistalPhalanxTW',
                 'Earthquakes',
                 'ECG200',
                 'ECG5000',
                 'ECGFiveDays',
                 'ElectricDevices',
                 'FaceAll']
    elif args.dataset == "debug1":
        flist = ['FaceFour',
                 'FacesUCR',
                 'FISH',
                 'FordA',
                 'FordB',
                 'Gun_Point',
                 'Ham',
                 'HandOutlines',
                 'Haptics',
                 'Herring',
                 'InlineSkate',
                 'InsectWingbeatSound']
    elif args.dataset == "debug2":
        flist = ['ItalyPowerDemand',
                 'LargeKitchenAppliances',
                 'Lighting2',
                 'Lighting7',
                 'MALLAT',
                 'Meat',
                 'MedicalImages',
                 'MiddlePhalanxOutlineAgeGroup',
                 'MiddlePhalanxOutlineCorrect',
                 'MiddlePhalanxTW',
                 'MoteStrain',
                 'NonInvasiveFatalECG_Thorax1',
                 'NonInvasiveFatalECG_Thorax2']
    elif args.dataset == "debug3":
        flist = ['OliveOil',
                 'OSULeaf',
                 'PhalangesOutlinesCorrect',
                 'Phoneme',
                 'Plane',
                 'ProximalPhalanxOutlineAgeGroup',
                 'ProximalPhalanxOutlineCorrect',
                 'ProximalPhalanxTW',
                 'RefrigerationDevices',
                 'ScreenType']
    elif args.dataset == "debug4":
        flist = ['ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'SonyAIBORobotSurface',
                 'SonyAIBORobotSurfaceII',
                 'StarLightCurves',
                 'Strawberry',
                 'SwedishLeaf',
                 'Symbols',
                 'synthetic_control']
    elif args.dataset == "debug5":
        flist = ['ToeSegmentation1',
                 'ToeSegmentation2',
                 'Trace',
                 'Two_Patterns',
                 'TwoLeadECG',
                 'uWaveGestureLibrary_X',
                 'uWaveGestureLibrary_Y',
                 'uWaveGestureLibrary_Z',
                 'UWaveGestureLibraryAll',
                 'wafer',
                 'Worms',
                 'WormsTwoClass',
                 'yoga']
    elif args.dataset == "debug6":
        flist = ['OliveOil',
                 'SwedishLeaf',
                 'Symbols',
                 'synthetic_control',
                 'ToeSegmentation1',
                 'ToeSegmentation2',
                 'Worms',
                 'WormsTwoClass',
                 'yoga',
                 'Trace',
                 'Two_Patterns',
                 'TwoLeadECG',
                 'uWaveGestureLibrary_X',
                 'uWaveGestureLibrary_Y',
                 'uWaveGestureLibrary_Z',
                 'UWaveGestureLibraryAll',
                 'wafer',
                 'ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'SonyAIBORobotSurface',
                 'SonyAIBORobotSurfaceII',
                 'StarLightCurves',
                 'Strawberry',
                 'OSULeaf',
                 'PhalangesOutlinesCorrect',
                 'Phoneme',
                 'Plane',
                 'ProximalPhalanxOutlineAgeGroup',
                 'ProximalPhalanxOutlineCorrect',
                 'ProximalPhalanxTW',
                 'RefrigerationDevices',
                 'ScreenType'
                 ]
    elif args.dataset == "debug7":
        flist = ['FordB',
                 'Gun_Point',
                 'Ham',
                 'HandOutlines',
                 'Haptics',
                 'Herring',
                 'InlineSkate',
                 'InsectWingbeatSound',
                 'NonInvasiveFatalECG_Thorax1',
                 'NonInvasiveFatalECG_Thorax2',
                 'PhalangesOutlinesCorrect',
                 'Phoneme',
                 'ProximalPhalanxOutlineAgeGroup',
                 'RefrigerationDevices',
                 'ScreenType',
                 'ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'SonyAIBORobotSurface',
                 'StarLightCurves',
                 'Strawberry',
                 'Symbols',
                 'synthetic_control',
                 'Trace',
                 'Two_Patterns',
                 'uWaveGestureLibrary_X',
                 'uWaveGestureLibrary_Y',
                 'uWaveGestureLibrary_Z',
                 'UWaveGestureLibraryAll',
                 'wafer',
                 'Worms',
                 'WormsTwoClass',
                 'yoga'
                 ]
    elif args.dataset == "debug8":
        flist = ["SwedishLeaf"]
    elif args.dataset == "debug9conv1-100":
        flist = [
            'HandOutlines',
            'Haptics',
            'InlineSkate',
            'Phoneme',
            'ProximalPhalanxOutlineAgeGroup',
            'ScreenType',
            'ShapeletSim',
            'ShapesAll',
            'SonyAIBORobotSurface',
            'StarLightCurves',
            'Strawberry',
            'uWaveGestureLibrary_Z',
        ]
    elif args.dataset == "debug9conv1-100-reverse":
        flist = [
            'HandOutlines',
            'Haptics',
            'InlineSkate',
            'Phoneme',
            'ProximalPhalanxOutlineAgeGroup',
            'ScreenType',
            'ShapeletSim',
            'ShapesAll',
            'SonyAIBORobotSurface',
            'StarLightCurves',
            'Strawberry',
        ]
        flist = reversed(flist)
    elif args.dataset == "debug10conv1-99":
        flist = ['Haptics',
                 'Herring',
                 'InlineSkate',
                 'InsectWingbeatSound',
                 'LargeKitchenAppliances',
                 'Lighting2',
                 'MALLAT',
                 'NonInvasiveFatalECG_Thorax1',
                 'NonInvasiveFatalECG_Thorax2',
                 'OliveOil',
                 'Phoneme',
                 'RefrigerationDevices',
                 'ScreenType',
                 'ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'StarLightCurves',
                 'UWaveGestureLibraryAll',
                 'Worms',
                 'WormsTwoClass',  # start from almost the beginning
                 'Earthquakes'
                 ]
    elif args.dataset == "debug10conv1-99-reverse":
        flist = ['HandOutlines',
                 'Haptics',
                 'InlineSkate',
                 'LargeKitchenAppliances',
                 'Lighting2',
                 'MALLAT',
                 'NonInvasiveFatalECG_Thorax1',
                 'NonInvasiveFatalECG_Thorax2',
                 'OliveOil',
                 'Phoneme',
                 'RefrigerationDevices',
                 'ScreenType',
                 'ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'StarLightCurves',
                 'UWaveGestureLibraryAll',
                 'Worms',
                 'WormsTwoClass',  # start from almost the beginning
                 'Earthquakes'
                 ]
        flist = reversed(flist)
    else:
        raise AttributeError("Unknown dataset: ", args.dataset)

    if torch.cuda.is_available():
        print("CUDA is available")
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available")

    # flist = sorted(flist, key=lambda s: s.lower())
    # flist = flist[3:]  # start from Beef
    # reversed(flist)
    print("flist: ", flist)
    preserve_energies = args.preserve_energies
    for dataset_name in flist:
        args.dataset_name = dataset_name
        for preserve_energy in preserve_energies:
            print("Dataset: ", dataset_name)
            print("preserve energy: ", preserve_energy)
            args.preserve_energy = preserve_energy
            start_training = time.time()
            try:
                main(args=args)
            except RuntimeError as err:
                print(f"ERROR: {dataset_name}. "
                      "Details: " + str(err))
                traceback.print_tb(err.__traceback__)
            print("training time (sec): ", time.time() - start_training)

    print("total elapsed time (sec): ", time.time() - start_time)
