#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
"""

import os
import sys
import pathlib
import logging
import traceback

import numpy as np
import socket
import time
import shutil
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch
from torch.optim.lr_scheduler import MultiStepLR

from cnns.nnlib.pytorch_layers.AdamFloat16 import AdamFloat16
from cnns.nnlib.pytorch_architecture.model_utils import getModelPyTorch
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
from cnns.nnlib.utils.general_utils import PrecisionType
from cnns.nnlib.utils.general_utils import Bool
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import mem_log_file
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.mnist import get_mnist
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.ucr.ucr import get_ucr
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.pytorch_experiments.utils.progress_bar import progress_bar

# from memory_profiler import profile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]

"""
sources:
https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/
"""
try:
    import apex
except ImportError:
    raise ImportError("""Please install apex from 
    https://www.github.com/nvidia/apex to run this code.""")

amp_handle = None
# from apex.parallel import DistributedDataParallel as DDP
# from apex.fp16_utils import input_to_half
from apex.fp16_utils import network_to_half
from apex.fp16_utils import FP16_Optimizer

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

args = get_args()

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

if torch.cuda.is_available() and args.use_cuda:
    print("cuda is available: ")
    device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")


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

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # fp16 (apex) - the data is cast explicitely to fp16 via data.to() method.
        data, target = data.to(device=device, dtype=args.dtype), target.to(
            device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        # The cross entropy loss combines `log_softmax` and `nll_loss` in
        # a single function.
        # loss = F.cross_entropy(output, target)

        if args.precision_type is PrecisionType.AMP:
            """
            https://github.com/NVIDIA/apex/tree/master/apex/amp
            
            Not used: at each optimization step in the training loop, 
            perform the following:
            Cast gradients to FP32. If a loss was scaled, descale the 
            gradients. Apply updates in FP32 precision and copy the updated 
            parameters to the model, casting them to FP16.
            """
            with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss = optimizer.scale_loss(loss)

        elif args.precision_type is PrecisionType.FP16:
            optimizer.backward(loss)
        elif args.precision_type is PrecisionType.FP32:
            loss.backward()
        else:
            raise Exception(
                f"Unsupported precision type for float16: {args.precision_type}")

        optimizer.step()

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100.0 * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if args.log_conv_size is True:
            with open(additional_log_file, "a") as file:
                file.write("\n")

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
            test_loss += loss_function(output,
                                       target).item()  # sum up batch loss
            # get the index of the max log-probability
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if args.log_conv_size is True:
                with open(additional_log_file, "a") as file:
                    file.write("\n")

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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# @profile
def main(args):
    """
    The main training.

    :param dataset_name: the name of the dataset from UCR.
    """
    is_debug = args.is_debug
    dataset_name = args.dataset_name
    preserve_energy = args.preserve_energy
    compress_rate = args.compress_rate

    dataset_log_file = os.path.join(
        results_folder_name, get_log_time() + "-dataset-" + str(dataset_name) + \
                             "-preserve-energy-" + str(preserve_energy) + \
                             "-compress-rate-" + str(compress_rate) + \
                             ".log")
    DATASET_HEADER = HEADER + ",dataset," + str(dataset_name) + \
                     "-current-preserve-energy-" + str(preserve_energy) + "\n"
    with open(dataset_log_file, "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)
        # Write the header with the names of the columns.
        file.write(
            "epoch,train_loss,train_accuracy,dev_loss,dev_accuracy,test_loss,"
            "test_accuracy,epoch_time,learning_rate,train_time,test_time\n")

    # with open(os.path.join(results_dir, additional_log_file), "a") as file:
    #     # Write the metadata.
    #     file.write(DATASET_HEADER)

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

    train_loader, dev_loader, test_loader = None, None, None
    if dataset_name is "cifar10" or dataset_name is "cifar100":
        train_loader, test_loader, _, _ = get_cifar(args, dataset_name)
    elif dataset_name is "mnist":
        train_loader, test_loader = get_mnist(args)
    elif dataset_name in os.listdir(ucr_path):  # dataset from UCR archive
        train_loader, test_loader, dev_loader = get_ucr(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = getModelPyTorch(args=args)
    model.to(device)
    # model = torch.nn.DataParallel(model)

    # https://pytorch.org/docs/master/notes/serialization.html
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=device))
        msg = "loaded model: " + args.model_path
        # logger.info(msg)
        print(msg)
    if args.precision_type is PrecisionType.FP16:
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

    if args.precision_type is PrecisionType.FP16:
        """
        amp_handle: tells it where back-propagation occurs so that it can 
        properly scale the loss and clear internal per-iteration state.
        """
        # amp_handle = amp.init()
        # optimizer = amp_handle.wrap_optimizer(optimizer)

        # The optimizer supported by apex.
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   verbose=True)

    # max = choose the best model.
    min_train_loss = min_test_loss = min_dev_loss = sys.float_info.max
    max_train_accuracy = max_test_accuracy = max_dev_accuracy = 0.0

    # Optionally resume from a checkpoint.
    if args.resume:
        # Use a local scope to avoid dangling references.
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume,
                                        map_location=lambda storage,
                                                            loc: storage.cuda(
                                            args.gpu))
                args.start_epoch = checkpoint['epoch']
                max_train_accuracy = checkpoint['max_train_accuracy']
                model.load_state_dict(checkpoint['state_dict'])
                # An FP16_Optimizer instance's state dict internally stashes the master params.
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                return max_train_accuracy
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return 0.0

        max_train_accuracy = resume()

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
                    elapsed_time) + ",visualize," + str(
                    args.preserve_energy) + "," + str(
                    args.compress_rate) + "\n")
        return

    dataset_start_time = time.time()
    dev_loss = dev_accuracy = None
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        # print("\ntrain:")
        if args.log_conv_size is True:
            with open(additional_log_file, "a") as file:
                file.write(str(args.compress_rate) + ",")
        train_start_time = time.time()
        train_loss, train_accuracy = train(
            model=model, device=device, train_loader=train_loader, args=args,
            optimizer=optimizer, loss_function=loss_function, epoch=epoch)
        train_time = time.time() - train_start_time
        if args.is_dev_dataset:
            if dev_loader is None:
                raise Exception("The dev_loader was not set! Check methods to"
                                "get the data, e.g. get_ucr()")
            dev_loss, dev_accuracy = test(
                model=model, device=device, test_loader=dev_loader,
                loss_function=loss_function, args=args)
        # print("\ntest:")
        test_start_time = time.time()
        if args.log_conv_size is True:
            test_loss, test_accuracy = 0, 0
        else:
            test_loss, test_accuracy = test(
                model=model, device=device, test_loader=test_loader,
                loss_function=loss_function, args=args)
        test_time = time.time() - test_start_time
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start_time

        raw_optimizer = optimizer
        if args.precision_type is PrecisionType.FP16:
            raw_optimizer = optimizer.optimizer
        lr = f"unknown (started with: {args.learning_rate})"
        if len(raw_optimizer.param_groups) > 0:
            lr = raw_optimizer.param_groups[0]['lr']

        with open(dataset_log_file, "a") as file:
            file.write(str(epoch) + "," + str(train_loss) + "," + str(
                train_accuracy) + "," + str(dev_loss) + "," + str(
                dev_accuracy) + "," + str(test_loss) + "," + str(
                test_accuracy) + "," + str(epoch_time) + "," + str(
                lr) + "," + str(train_time) + "," + str(test_time) + "\n")

        # Metric: select the best model based on the best train loss (minimal).
        is_best = False
        if args.is_dev_dataset:
            if dev_accuracy > max_dev_accuracy:
                min_train_loss = train_loss
                max_train_accuracy = train_accuracy
                min_dev_loss = dev_loss
                max_dev_accuracy = dev_accuracy
                min_test_loss = test_loss
                max_test_accuracy = test_accuracy
                is_best = True
                model_path = os.path.join(models_dir,
                                          get_log_time() + "-dataset-" + str(
                                              dataset_name) + \
                                          "-preserve-energy-" + str(
                                              preserve_energy) + \
                                          "-compress-rate-" + str(
                                              args.compress_rate) + \
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
                is_best = True
                model_path = os.path.join(models_dir,
                                          get_log_time() + "-dataset-" + str(
                                              dataset_name) + \
                                          "-preserve-energy-" + str(
                                              preserve_energy) + \
                                          "-test-accuracy-" + str(
                                              test_accuracy) + ".model")
                torch.save(model.state_dict(), model_path)

        # Save the checkpoint (to resume training).
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'max_train_accuracy': max_train_accuracy,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename=dataset_name + "-" + str(
                max_train_accuracy) + "-" + str(
                args.compress_rate) + "-" + "checkpoint.tar")

    with open(global_log_file, "a") as file:
        file.write(dataset_name + "," + str(min_train_loss) + "," + str(
            max_train_accuracy) + "," + str(min_dev_loss) + "," + str(
            max_dev_accuracy) + "," + str(min_test_loss) + "," + str(
            max_test_accuracy) + "," + str(
            time.time() - dataset_start_time) + "\n")


if __name__ == '__main__':
    print("start learning!")
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
        file.write(HEADER + ",")
        file.write(
            "preserve_energies: " +
            ",".join(
                [str(energy) for energy in args.preserve_energies]) +
            ",")
        file.write(
            "compress_rates: " +
            ",".join(
                [str(compress_rate) for compress_rate in args.compress_rates]) +
            "\n")
        file.write(
            "dataset,"
            "min_train_loss,max_train_accuracy,"
            "min_dev_loss,max_dev_accuracy,"
            "min_test_loss,max_test_accuracy,"
            "execution_time,additional_info\n")

    if args.precision_type is PrecisionType.AMP:
        from apex import amp

        amp_handle = amp.init(enabled=True)

    if args.dataset == "all" or args.dataset == "ucr":
        flist = os.listdir(ucr_path)
    elif args.dataset == "cifar10":
        flist = ["cifar10"]
    elif args.dataset == "cifar100":
        flist = ["cifar100"]
    elif args.dataset == "mnist":
        flist = ["mnist"]
    elif args.dataset == "debug":
        # flist = ['ItalyPowerDemand']
        # flist = ['Lighting7']
        # flist = ['Trace']
        # flist = ["ToeSegmentation1"]
        # flist = ["Plane"]
        # flist = ["MiddlePhalanxOutlineAgeGroup"]
        # flist = ["ECGFiveDays"]
        # flist = ["MoteStrain"]
        # flist = ["Cricket_Y"]
        # flist = ["Strawberry"]
        # flist = ["FaceFour"]
        #         'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
        #         'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X',
        #         'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
        #         'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        #         'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000',
        #         'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
        #         'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham',
        #         'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
        #         'InsectWingbeatSound',
        # flist = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly',
        #          'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
        #          'CinC_ECG_torso', 'Coffee', 'Computers']
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
        flist = ['Computers', 'Beef', 'BeetleFly',
                 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
                 'CinC_ECG_torso', 'Coffee', 'Cricket_X',
                 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
                 'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000',
                 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                 'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham',
                 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
                 'InsectWingbeatSound', 'ItalyPowerDemand',
                 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT',
                 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
                 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
                 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
                 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
                 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
                 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
                 'SwedishLeaf', 'Symbols', 'synthetic_control',
                 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                 'Two_Patterns',
                 'TwoLeadECG', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                 'uWaveGestureLibrary_Z', 'UWaveGestureLibraryAll', 'wafer',
                 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga',
                 'ztest']
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
    elif args.dataset == "debug12":
        flist = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly',
                 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
                 'CinC_ECG_torso', 'Coffee', 'Computers']
    elif args.dataset == "debug13":
        flist = ['Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass']
    elif args.dataset == "debug14":
        flist = ['MALLAT', 'Meat', 'MedicalImages',
                 'MiddlePhalanxOutlineAgeGroup']
    elif args.dataset == "debug15":
        flist = ['Coffee', 'Computers', 'Phoneme', 'Plane', 'Car', 'Strawberry']
    elif args.dataset == "debug16":
        flist = ['Lighting7', 'FISH', 'FaceFour',
                 'ProximalPhalanxOutlineCorrect']
    elif args.dataset == "debug17":
        flist = ['MoteStrain', 'ECGFiveDays', 'DistalPhalanxOutlineAgeGroup',
                 'ProximalPhalanxTW']
    elif args.dataset == "debug18":
        flist = ["Cricket_X", "Cricket_Y", "Cricket_Z"]
    elif args.dataset == "debug19":
        flist = ["Strawberry", "Beef", 'ScreenType',
                 'ShapeletSim',
                 'ShapesAll',
                 'SmallKitchenAppliances',
                 'StarLightCurves',
                 'UWaveGestureLibraryAll',
                 'Worms',
                 'WormsTwoClass',  # start from almost the beginning
                 'Earthquakes']
    elif args.dataset == 'debug20':
        flist = ["Coffee", "Car", "ArrowHead"]
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
    for dataset_name in flist:
        args.dataset_name = dataset_name
        for preserve_energy in args.preserve_energies:
            for compress_rate in args.compress_rates:
                print("Dataset: ", dataset_name)
                print("preserve energy: ", preserve_energy)
                print("compress rate: ", compress_rate)
                args.preserve_energy = preserve_energy
                args.compress_rate = compress_rate
                start_training = time.time()
                try:
                    main(args=args)
                except RuntimeError as err:
                    print(f"ERROR: {dataset_name}. "
                          "Details: " + str(err))
                    traceback.print_tb(err.__traceback__)
                print("training time (sec): ", time.time() - start_training)

    print("total elapsed time (sec): ", time.time() - start_time)
