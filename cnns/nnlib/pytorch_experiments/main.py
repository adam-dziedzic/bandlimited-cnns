#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
"""
from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)

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

from cnns.nnlib.pytorch_experiments.utils.optim_utils import get_optimizer
from cnns.nnlib.pytorch_experiments.utils.optim_utils import get_loss_function
from cnns.nnlib.pytorch_experiments.utils.optim_utils import get_scheduler
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import PrecisionType
from cnns.nnlib.utils.general_utils import AttackType
from cnns.nnlib.utils.general_utils import PredictionType
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import mem_log_file
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.mnist.mnist import get_mnist
from cnns.nnlib.datasets.synthetic.synthetic import get_synthetic
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.svhn import get_svhn
from cnns.nnlib.datasets.ucr.ucr import get_ucr
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.utils.exec_args import get_args
# from cnns.nnlib.pytorch_experiments.track_utils.progress_bar import progress_bar
from cnns.nnlib.pytorch_architecture.get_model_architecture import \
    getModelPyTorch
from cnns.nnlib.pytorch_experiments.utils.progress_bar import progress_bar
# from memory_profiler import profile
from cnns.nnlib.datasets.deeprl.rollouts import get_rollouts_dataset
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_batch
from cnns.nnlib.utils.general_utils import NetworkType

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]

"""
sources:
https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/
"""
args = get_args()

if torch.cuda.is_available() and args.use_cuda and args.precision_type is PrecisionType.AMP:
    try:
        import apex

        amp_handle = None
        # from apex.parallel import DistributedDataParallel as DDP
        # from apex.fp16_utils import input_to_half
        from apex.fp16_utils import network_to_half
        from apex.fp16_utils import FP16_Optimizer
    except ImportError:
        raise ImportError("""Please install apex from 
        https://www.github.com/nvidia/apex to run this code.""")

dir_path = os.path.dirname(os.path.realpath(__file__))
# print("current working directory: ", dir_path)

ucr_data_folder = "TimeSeriesDatasets"
# ucr_path = os.path.join(dir_path, os.pardir, data_folder)
ucr_path = os.path.join(os.pardir, ucr_data_folder)

results_folder_name = "results"
results_dir = os.path.join(os.getcwd(), results_folder_name)
# print("current dir: ", os.getcwd())
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

models_folder_name = "models"
models_dir = os.path.join(os.getcwd(), models_folder_name)
# print("models_dir: ", models_dir)
pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

# plt.switch_backend('agg')

current_file_name = __file__.split("/")[-1].split(".")[0]


# print("current file name: ", current_file_name)

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
def train(model, train_loader, optimizer, loss_function, args, epoch=None):
    """
    Train the model.

    :param model: deep learning model.
    :param train_loader: the training dataset.
    :param optimizer: Adam, Momentum, etc.
    :param epoch: the current epoch number.
    """

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # fp16 (apex) - the data is cast explicitely to fp16 via data.to() method.
        optimizer.zero_grad()

        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device=args.device,
                               dtype=args.dtype)
            output = model(data)
        else:
            data = data.to(device=args.device, dtype=args.dtype)
            output = model(data)

        target = target.to(device=args.device)

        # if args.svd_transform > 0.0:
        #     compress_svd_batch(x=data, compress_rate=args.svd_transform)

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

        total += target.size(0)

        if args.prediction_type == PredictionType.CLASSIFICATION:
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

        if args.log_conv_size is True:
            with open(additional_log_file, "a") as file:
                file.write("\n")

        # if args.is_progress_bar:
        #     progress_bar(total, len(train_loader.dataset), epoch=epoch,
        #                  msg="Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)" %
        #                      (train_loss / total, 100. * correct / total,
        #                       correct,
        #                       total))

    # Test loss for the whole dataset.
    train_loss /= total
    accuracy = 100. * correct / total

    return train_loss, accuracy


def test(model, test_loader, loss_function, args, epoch=None):
    """
    Test the model and return test loss and accuracy.

    :param model: deep learning model.
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
            if isinstance(data, dict):
                for k, v in data.items():
                    data[k] = v.to(device=args.device,
                                   dtype=args.dtype)
                output = model(data)
            else:
                data = data.to(device=args.device, dtype=args.dtype)
                output = model(data)

            target = target.to(args.device)

            # if args.svd_transform > 0.0:
            #     compress_svd_batch(x=data, compress_rate=args.svd_transform)

            # sum up batch loss
            test_loss += loss_function(output, target.squeeze()).item()

            total += target.size(0)

            if args.prediction_type == PredictionType.CLASSIFICATION:
                # get the index of the max log-probability
                _, predicted = output.max(1)
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
    dataset_name = args.dataset_name
    preserve_energy = args.preserve_energy
    compress_rate = args.compress_rate

    DATASET_HEADER = HEADER + ",dataset," + str(dataset_name) + \
                     "-current-preserve-energy-" + str(preserve_energy) + "\n"

    if args.test_compress_rates:
        dataset_log_file = os.path.join(results_folder_name,
                                        f"{args.dataset}-dataset-compress-rates.log")
    else:
        dataset_log_file = os.path.join(
            results_folder_name,
            get_log_time() + "-dataset-" + str(dataset_name) + \
            "-preserve-energy-" + str(preserve_energy) + \
            "-compress-rate-" + str(compress_rate) + \
            ".log")
        with open(dataset_log_file, "a") as file:
            # Write the metadata.
            file.write(DATASET_HEADER)
            # Write the header with the names of the columns.
            header = ['epoch',
                      'train_accruacy',
                      'test_accuracy',
                      'epoch_time',
                      'train_loss',
                      'test_loss',
                      'dev_loss',
                      'dev_accuracy',
                      'learning_rate',
                      'train_time',
                      'test_time',
                      'compress_rate'
                      ]
            header = args.delimiter.join(header)
            file.write(header + '\n')
            print(header)

    # with open(os.path.join(results_dir, additional_log_file), "a") as file:
    #     # Write the metadata.
    #     file.write(DATASET_HEADER)

    with open(os.path.join(results_dir, mem_log_file), "a") as file:
        # Write the metadata.
        file.write(DATASET_HEADER)

    torch.manual_seed(args.seed)

    use_cuda = args.use_cuda
    tensor_type = args.tensor_type
    if use_cuda and args.noise_sigma is False:
        if tensor_type is TensorType.FLOAT32:
            cuda_type = torch.cuda.FloatTensor
        elif tensor_type is TensorType.FLOAT16:
            cuda_type = torch.cuda.HalfTensor
        elif tensor_type is TensorType.DOUBLE:
            cuda_type = torch.cuda.DoubleTensor
        else:
            raise Exception(f"Unknown tensor type: {tensor_type}")
        # The below has to be disabled for normal distribution to work (add noise).
        torch.set_default_tensor_type(cuda_type)
    elif use_cuda is False and args.noise_sigma is False:
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
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        train_loader, test_loader, _, _ = get_cifar(args, dataset_name)
    elif dataset_name == "mnist" or dataset_name == "mnist_svd":
        train_loader, test_loader, _, _ = get_mnist(args)
    elif dataset_name == "synthetic":
        train_loader, test_loader, _, _ = get_synthetic(args)
    elif dataset_name == "imagenet":
        train_loader, test_loader, _, _ = load_imagenet(args)
    elif dataset_name == "svhn":
        train_loader, test_loader, _, _ = get_svhn(args)
    elif "WIFI" in dataset_name or dataset_name.startswith(
            '2_classes_WiFi') or dataset_name.startswith(
        'Case') or dataset_name.startswith('2_classes_WIFI'):
        # train_loader, test_loader, dev_loader = get_ucr(args)
        test_loader, train_loader, dev_loader = get_ucr(args)
    elif dataset_name in os.listdir(ucr_path):  # dataset from UCR archive
        train_loader, test_loader, dev_loader = get_ucr(args)
    elif dataset_name == "deeprl":
        train_loader, test_loader, _, _ = get_rollouts_dataset(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = getModelPyTorch(args=args)
    model.to(args.device)
    # model = torch.nn.DataParallel(model)

    # https://pytorch.org/docs/master/notes/serialization.html
    if args.model_path != "no_model" and args.model_path != "pretrained":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=args.device))
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

    optimizer = get_optimizer(args=args, model=model)

    scheduler = get_scheduler(args=args, optimizer=optimizer)

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

    loss_function = get_loss_function(args)

    if args.visulize is True:
        start_visualize_time = time.time()

        train_loss, train_accuracy = 0, 0
        # train_loss, train_accuracy = test(
        #     model=model, test_loader=train_loader,
        #     loss_function=loss_function, args=args)

        test_loss, test_accuracy = test(
            model=model, test_loader=test_loader,
            loss_function=loss_function, args=args)

        elapsed_time = time.time() - start_visualize_time

        print("test time: ", elapsed_time)
        print("test accuracy: ", test_accuracy)
        print("train accuracy: ", train_accuracy)

        with open(global_log_file, "a") as file:
            file.write(
                dataset_name + "," + str(train_loss) + "," + str(
                    train_accuracy) + ",None,None," + str(
                    test_loss) + "," + str(test_accuracy) + "," + str(
                    elapsed_time) + ",visualize," + str(
                    args.preserve_energy) + "," + str(
                    args.compress_rate) + "\n")
        return

    dataset_start_time = time.time()
    dev_loss = min_dev_los = sys.float_info.max
    dev_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        # print("\ntrain:")
        if args.log_conv_size is True:
            with open(additional_log_file, "a") as file:
                file.write(str(args.compress_rate) + ",")
        train_start_time = time.time()
        train_loss, train_accuracy = train(
            model=model, train_loader=train_loader,
            args=args,
            optimizer=optimizer, loss_function=loss_function, epoch=epoch)
        train_time = time.time() - train_start_time
        if args.is_dev_dataset:
            if dev_loader is None:
                raise Exception("The dev_loader was not set! Check methods to"
                                "get the data, e.g. get_ucr()")
            dev_loss, dev_accuracy = test(
                model=model, test_loader=dev_loader,
                loss_function=loss_function, args=args)
        # print("\ntest:")
        test_start_time = time.time()
        if args.log_conv_size is True or args.mem_test is True:
            test_loss, test_accuracy = 0, 0
        else:
            test_loss, test_accuracy = test(
                model=model, test_loader=test_loader,
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
            msg = [epoch,
                   train_accuracy,
                   test_accuracy,
                   epoch_time,
                   train_loss,
                   test_loss,
                   dev_loss,
                   dev_accuracy,
                   lr,
                   train_time,
                   test_time,
                   args.compress_rate]
            msg = args.delimiter.join([str(x) for x in msg])
            print(msg)
            file.write(msg + "\n")

        # Metric: select the best model based on the best train loss (minimal).
        is_best = False
        if (epoch == args.start_epoch) or (train_loss < min_train_loss) or (
                dev_loss < min_dev_loss):
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
                                      "-test-loss-" + str(
                                          test_loss) + \
                                      "-test-accuracy-" + str(
                                          test_accuracy) +
                                      "-channel-vals-" + str(
                                          args.values_per_channel) + \
                                      "-env_name-" + args.env_name + \
                                      ".model")
            torch.save(model.state_dict(), model_path)

        # Save the checkpoint (to resume training).
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'max_train_accuracy': max_train_accuracy,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best,
        #     filename="dataset-" + dataset_name + "-max-train-accuracy-" + str(
        #         max_train_accuracy) + "-max-test-accuracy-" + str(
        #         max_test_accuracy) + "-compress-rate-" + str(
        #         args.compress_rate) + "-" + "checkpoint.tar")

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
    try:
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        cuda_visible_devices = 0

    global_log_file = os.path.join(results_folder_name,
                                   get_log_time() + "-ucr-fcnn.log")
    args_str = args.get_str()
    print('args: ', args_str)
    HEADER = "hostname," + str(
        hostname) + ",timestamp," + get_log_time() + "," + str(
        args_str) + ",cuda_visible_devices," + str(cuda_visible_devices)
    with open(additional_log_file, "a") as file:  # Write the metadata.
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
        flist = sorted(os.listdir(ucr_path))
    elif args.dataset == "reverse-ucr":
        flist = reversed(sorted(os.listdir(ucr_path)))
    elif args.dataset == "cifar10":
        flist = ["cifar10"]
    elif args.dataset == "cifar100":
        flist = ["cifar100"]
    elif args.dataset == "mnist":
        flist = ["mnist"]
    elif args.dataset == "mnist_svd":
        flist = ["mnist"]
    elif args.dataset == "synthetic":
        flist = ["synthetic"]
    elif args.dataset == "svhn":
        flist = ["svhn"]
    elif args.dataset == "imagenet":
        flist = ["imagenet"]
    elif args.dataset == "deeprl":
        flist = ["deeprl"]
    elif "WIFI" in args.dataset or args.dataset.startswith(
            '2_classes_WiFi') or args.dataset.startswith('Case'):
        flist = [args.dataset]
    else:
        raise AttributeError("Unknown dataset: ", args.dataset)

    if torch.cuda.is_available():
        print("CUDA is available")
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available")

    print("flist: ", flist)
    print('ucr path: ', args.ucr_path)
    compress_rate = 0
    for compress_rate in args.compress_rates:
        # for noise_epsilon in args.noise_epsilons:
        # args.noise_epsilon = noise_epsilon
        print("compress rate: ", compress_rate)

        # This is to run many experiments and get a single file with answers.
        # This assumes that we use only a single additional laver for the
        # ResNet network.
        args.compress_rate = compress_rate

        # compression techniques
        if args.attack_type == AttackType.GAUSS_ONLY:
            args.compress_fft_layer = compress_rate
        if args.attack_type == AttackType.ROUND_ONLY:
            args.values_per_channel = compress_rate
        if args.attack_type == AttackType.SVD_ONLY:
            args.svd_compress = compress_rate

        for dataset_name in flist:
            args.dataset_name = dataset_name
            print("Dataset: ", dataset_name)
            for preserve_energy in args.preserve_energies:
                print("preserve energy: ", preserve_energy)
                args.preserve_energy = preserve_energy
                for noise_sigma in args.noise_sigmas:
                    print("noise sigma: ", noise_sigma)
                    args.noise_sigma = noise_sigma
                    for noise_epsilon in args.noise_epsilons:
                        args.noise_epsilon = noise_epsilon
                        for laplace_epsilon in args.laplace_epsilons:
                            args.laplace_epsilon = laplace_epsilon
                            for svd_transform in args.svd_compress_transform:
                                args.svd_transform = svd_transform
                                print('svd transform: ', svd_transform)
                                for fft_transform in args.fft_compress_transform:
                                    args.fft_transform = fft_transform
                                    print('fft transform: ', fft_transform)
                                    start_training = time.time()
                                    try:
                                        main(args=args)
                                    except RuntimeError as err:
                                        print(f"ERROR: {dataset_name}. "
                                              "Details: " + str(err))
                                        traceback.print_tb(err.__traceback__)
                                    print("elapsed time (sec): ",
                                          time.time() - start_training)

    print("total elapsed time (sec); ", time.time() - start_time)
