#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 17:20:19 2018
@author: ady@uchicago.edu
"""

import os
import time
import keras
import numpy as np
import torch.nn as nn
import torch
from torch.nn import AdaptiveAvgPool1d
from torch.nn.functional import softmax
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.utils import np_utils
import pandas as pd
import socket
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import get_log_time
import argparse
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)

nb_epochs = 2000

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

flist = ['Adiac']

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
parser.add_argument("-p", "--num_epochs", default=300, type=int,
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


def getModelKeras(input_size, nb_classes):
    """
    Create model.

    :param input_size: the length (width) of the time series.
    :param nb_classes: number of classes
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
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    model = Model(input=x, output=out)
    return model


class FCNNPytorch(nn.Module):

    def __init__(self, input_size, nb_classes, kernel_sizes=[8, 5, 3],
                 out_channels=[128, 256, 128],
                 strides=[1, 1, 1]):
        """
        Create the FCNN model in PyTorch.

        :param input_size: the length (width) of the time series.
        :param nb_classes: number of output classes.
        :param kernel_sizes: the sizes of the kernesl in each conv layer.
        :param out_channels: the number of filters for each conv layer.
        :param stride: the stride for the convolutions.
        """
        super(FCNNPytorch, self).__init__()
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.strides = strides

        self.relu = nn.ReLU(inplace=True)
        # For the "same" mode for the convolution, pad the input.
        conv_pads = [kernel_size - 1 for kernel_size in kernel_sizes]

        self.conv0 = nn.Conv1d(
            in_channels=1, out_channels=out_channels[0], stride=strides[0],
            kernel_size=kernel_sizes[0], padding=(0, conv_pads[0]))
        self.bn0 = nn.BatchNorm1d(num_features=out_channels[0])

        self.conv1 = nn.Conv1d(
            in_channels=out_channels[0], out_channels=out_channels[1],
            kernel_size=kernel_sizes[1], adding=(0, conv_pads[1]),
            stride=strides[1])
        self.bn1 = nn.BatchNorm1d(num_features=out_channels[1])

        self.conv2 = nn.Conv1d(
            in_channels=out_channels[1], out_channels=out_channels[2],
            kernel_size=kernel_sizes[2], adding=(0, conv_pads[2]),
            stride=strides[2])
        self.bn2 = nn.BatchNorm1d(num_features=out_channels[2])

        self.avg = AdaptiveAvgPool1d(input_size)

        self.lin = nn.Linear(input_size, nb_classes)

    def forward(self, x):
        """
        The forward pass through the network.

        :param x: the input data for the network.
        :return: the output class.
        """

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avg(out)
        out = self.lin(out)

        out = softmax(out)

        return out


def getModelPyTorch(input_size, nb_classes):
    """
    Get the PyTorch version of the FCNN model.

    :param input_size: the length (width) of the time series.
    :param nb_classes: number of output classes.

    :return: the model.
    """
    return FCNNPytorch(input_size=input_size, nb_classes=nb_classes)


def getData(fname):
    x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] // 10, 16)

    y_train = (y_train - y_train.min()) // (y_train.max() - y_train.min()) * (
            nb_classes - 1)
    y_test = (y_test - y_test.min()) // (y_test.max() - y_test.min()) * (
            nb_classes - 1)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) // (x_train_std)

    x_test = (x_test - x_train_mean) // (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test, batch_size, nb_classes


def run_keras():
    for each in flist:
        fname = each

        x_train, y_train, x_test, y_test, batch_size, nb_classes = getData(
            fname=fname)

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        model = getModelKeras(input_size=x_train.shape[1:],
                              nb_classes=nb_classes)

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
        # Print the testing results which has the lowest training loss.
        log = pd.DataFrame(hist.history)
        print(log.loc[log['loss'].idxmin]['loss'],
              log.loc[log['loss'].idxmin]['val_acc'])


def run_pytorch():
    for each in flist:
        fname = each

        x_train, y_train, x_test, y_test, batch_size, nb_size = getData(
            fname=fname)
        W = x_train.shape[1]  # The width of the time-series data.

        model = getModelPyTorch(input_size=W)



def train_network(net, trainloader, optimizer, criterion,
                  device=torch.device("cpu")):
    """
    Train the network

    Loop over the data iterator, and feed the inputs to the network and
    optimize.
    """
    for epoch in range(1):  # loop over the dataset once at a time

        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            # get the inputs
            inputs, labels = data
            # print("labels: ", labels)
            # move them to CUDA (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            start = time.time()

            # zero the parameter gradients before computing gradient for the
            # new batch
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.item()

    return time.time() - start, running_loss


def train_network_mem_test(net, trainloader, optimizer, criterion, batch_size,
                           input_size, device=torch.device("cpu")):
    """
    Train the network

    Loop over the data iterator, and feed the inputs to the network and
    optimize.
    """
    # optimization
    iter_number_print = iter_number_total
    forward_time = []
    backward_time = []
    optimizer_time = []
    total_time = []
    aggregator = lambda array: np.median(array, overwrite_input=False)

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            # get the inputs
            inputs, labels = data
            # print("labels: ", labels)
            # move them to CUDA (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            if net.input_channel != 3:
                # print("input channel: ", net.input_channel)
                # shrink to 1 layer and then expand to the required number of
                # channels
                # inputs = torch.tensor(inputs[:, 0:1, ...].expand(-1, net.input_channel, -1, -1))
                # generate the random data of required image size and the number of channels
                inputs = torch.randn(batch_size, net.input_channel,
                                     net.img_size_to_features,
                                     net.img_size_to_features).to(device)

            start_total = time.time()

            # zero the parameter gradients before computing gradient for the
            # new batch
            optimizer.zero_grad()

            # forward
            start = time.time()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            forward_time.append(time.time() - start)

            # backward
            start = time.time()
            loss.backward()
            backward_time.append(time.time() - start)

            # optimize
            start = time.time()
            optimizer.step()
            optimizer_time.append(time.time() - start)

            # print statistics
            running_loss += loss.item()

            total_time = time.time() - start_total

            if i % iter_number_print == iter_number_print - 1:  # print every 1 mini-batch
                print(
                    '[%d, %5d],forward time,%f,backward_time,%f,'
                    'optimizer_time,%f,total_time,%f,loss,%.3f,'
                    'input_size,%d,img_size,%d,batch_size,%d' %
                    (
                        epoch + 1, i + 1, aggregator(forward_time),
                        aggregator(backward_time),
                        aggregator(optimizer_time), aggregator(total_time),
                        running_loss / iter_number_print,
                        input_size, inputs.shape[-1], batch_size))
                running_loss = 0.0
            if i + 1 == iter_number_total:
                break

    return aggregator(forward_time), aggregator(backward_time), aggregator(
        optimizer_time), aggregator(total_time)


def imshow(img):
    """
    Show an image.

    :param img: image to show
    :return: nothing
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def test_network(net, testloader, classes, device):
    """
    ExperimentSpectralSpatial the network on the testing set.

    We have trained the network over the training dataset.
    Check what the network has learnt.

    We will check this by predicting the class label that the neural network
    outputs, and checking it against the ground-truth. If the prediction is
    correct, we add the sample to the list of correct predictions.

    :param testloader: an instance of the torch.utils.data.DataLoader
    :param classes: the classes expected in the test set
    :param net: the deep neural network model
    :return: the accuracy on the whole test set
    """

    # print images

    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ',
    #       ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Let us see what the neural network thinks these examples above are:

    # outputs = net(images)

    ########################################################################
    # The outputs are energies for the 10 classes.
    # Higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    # _, predicted = torch.max(outputs, 1)
    #
    # print('Predicted: ',
    #       ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total

    if is_debug:
        # what are the classes that performed well, and the classes that did
        # not perform well:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            if class_total[i] > 0:
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

    return final_accuracy


def get_optimizer(net, optimizer_type, weight_decay=0.0001, momentum=0.9,
                  lr=0.1):
    # Wrap model for multi-GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()

    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr,
                                momentum=momentum, nesterov=True,
                                weight_decay=weight_decay)
    if optimizer_type is OptimizerType.ADAM:
        optimizer = optim.Adam(params=net.parameters(),
                               weight_decay=weight_decay)

    milestones = [
        0.4 * num_epochs,
        0.6 * num_epochs,
        0.8 * num_epochs,
        0.9 * num_epochs]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    return optimizer, scheduler, net


def main():
    input_size = start_size  # cifar-10 images are 32 x 32 pixels
    # adjust the batch size to fully utilize the gpu
    batch_size = init_batch_size
    # if we are on a CUDA machine, then this should print a CUDA device
    # (otherwise it prints cpu):
    print("Currently used device: ", device)

    trainloader, testloader, classes = load_data_CIFAR10(input_size=input_size,
                                                         batch_size=batch_size)

    net = define_net(num_classes=len(classes), input_size=input_size,
                     batch_size=batch_size)
    net.to(device)

    # Define a Loss function
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler, net = get_optimizer(net, optimizer_type)

    total_time = 0.0
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        epoch_time, running_loss = train_network(
            net=net, optimizer=optimizer, criterion=criterion,
            trainloader=trainloader, device=device)
        total_time += epoch_time
        train_accuracy = test_network(net=net, testloader=trainloader,
                                      classes=classes, device=device)
        test_accuracy = test_network(net=net, testloader=testloader,
                                     classes=classes, device=device)

        with open(log_file, "a") as file:
            file.write(
                "timestamp," + str(get_log_time()) + ",epoch," + str(epoch) +
                ",train_accuracy," + str(train_accuracy) + ",test_accuracy," +
                str(test_accuracy) + ",total_time," + str(total_time) +
                ",running_loss," + str(running_loss) + "\n")


def plot_figure(batch_forward_times, batch_backward_times, batch_total_times,
                batch_input_sizes, batch_sizes,
                iter_number_total):
    fig, ax = plt.subplots()
    for batch_index, batch_size in enumerate(batch_sizes):
        input_sizes = batch_input_sizes[batch_index]
        forward_times = batch_forward_times[batch_index]
        backward_times = batch_backward_times[batch_index]
        total_times = batch_total_times[batch_index]

        # forward_label = "forward pass for batch size " + str(batch_size)
        # ax.plot(input_sizes, forward_times, label=forward_label)
        # backward_label = "backward pass for batch size " + str(batch_size)
        # ax.plot(input_sizes, backward_times, label=backward_label)
        total_label = "forward and backward pass (total) for batch size " + str(
            batch_size)
        ax.plot(input_sizes, total_times, label=total_label)

    ax.legend()
    input_sizes_example = batch_input_sizes[0]
    # fig.set_xticklables(input_sizes_example)

    ax.set_yscale('log')
    ax.set_xscale('log')

    # plt.xticks(input_sizes_example)

    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    # plt.title('Compare execution time of forward and backward passes')
    # plt.suptitle('iteration number per training: ' + str(iter_number_total), fontsize=12)
    plt.xlabel('Input size (size x size image)')
    plt.ylabel('Execution time (sec)')
    file_name = "Graphs/mem-forward-backward-" + current_file_name + "-" + get_log_time() + "-iterations-" + str(
        iter_number_total)
    plt.gcf().subplots_adjust(left=0.1)
    # plt.gcf().subplots_adjust(right=0.60)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.savefig(file_name + ".pdf")
    with open(file_name, "w+") as f:
        f.write("batch_sizes=")
        f.write(str(batch_sizes))
        f.write("\nbatch_forward_times=")
        f.write(str(batch_forward_times))
        f.write("\nbatch_backward_times=")
        f.write(str(batch_backward_times))
        f.write("\nbatch_input_sizes=")
        f.write(str(batch_input_sizes))
        f.write("\n")

    # plt.show()


def main_test():
    # If we are on a CUDA machine, then this should print a CUDA device
    # (otherwise it prints cpu).
    print("Currently used device: ", device)
    batch_sizes = []
    batch_forward_times = []
    batch_backward_times = []
    batch_optimizer_times = []
    batch_total_times = []
    batch_input_sizes = []

    batch_size = init_batch_size
    while batch_size <= max_batch_size:
        print("batch size: ", batch_size)
        forward_times = []
        backward_times = []
        optimizer_times = []
        total_times = []
        input_sizes = []

        try:
            input_size = start_size
            while input_size <= end_size:
                # print("input size: ", input_size)
                num_classes = 10
                net = define_net(num_classes=num_classes, input_size=input_size,
                                 batch_size=batch_size)
                net.to(device)

                trainloader, testloader, classes = load_data_CIFAR10(
                    input_size=input_size, batch_size=batch_size,
                    num_workers=num_workers,
                    channel_size=net.input_channel)

                assert num_classes == len(classes)

                # Define a Loss function and optimizer
                # Use a Classification Cross-Entropy loss and SGD with momentum.
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                criterion = nn.CrossEntropyLoss()

                forward_time, backward_time, optimizer_time, total_time = \
                    train_network_mem_test(
                        net=net, optimizer=optimizer,
                        criterion=criterion,
                        trainloader=trainloader,
                        batch_size=batch_size,
                        input_size=input_size,
                        device=device)

                forward_times.append(forward_time)
                backward_times.append(backward_time)
                optimizer_times.append(optimizer_time)
                total_times.append(total_time)

                input_sizes.append(input_size)
                input_size *= 2

                torch.cuda.empty_cache()
        # except RuntimeError as err:
        except NotImplementedError as err:
            print("Runtime error: " + str(err))

        batch_sizes.append(batch_size)
        batch_size *= 2

        batch_forward_times.append(forward_times)
        batch_backward_times.append(backward_times)
        batch_optimizer_times.append(optimizer_times)
        batch_total_times.append(total_times)
        batch_input_sizes.append(input_sizes)

        print("batch_sizes=", batch_sizes)
        print("batch_input_sizes=", batch_input_sizes)
        print("batch_forward_times=", batch_forward_times)
        print("batch_backward_times=", batch_backward_times)
        print("batch_optimizer_times=", batch_optimizer_times)
        print("batch_total_times=", batch_total_times)

    plot_figure(batch_forward_times=batch_forward_times,
                batch_backward_times=batch_backward_times,
                batch_total_times=batch_total_times,
                batch_input_sizes=batch_input_sizes, batch_sizes=batch_sizes,
                iter_number_total=iter_number_total)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    iter_number_total = args.iterations
    init_batch_size = args.initbatchsize
    max_batch_size = args.maxbatchsize
    start_size = args.startsize
    end_size = args.endsize
    num_workers = args.workers
    limit_size = args.limit_size
    conv_type = ConvType[args.conv_type]
    mem_test = args.mem_test
    num_epochs = args.num_epochs
    is_debug = args.is_debug
    optimizer_type = OptimizerType[args.optimizer_type]
    is_data_augmentation = args.is_data_augmentation
    device = args.device

    hostname = socket.gethostname()

    log_file = get_log_time() + "-" + str(conv_type) + ".log"
    with open(log_file, "a") as file:
        file.write(
            "hostname," + str(hostname) + ",timestamp," + get_log_time() +
            ",iter_number_total," + str(
                iter_number_total) + ",init_batch_size," + str(
                init_batch_size) + ",max_batch_size," + str(
                max_batch_size) + ",start_size," + str(
                start_size) + ",end_size," + str(
                end_size) + ",num_workers," + str(
                num_workers) + ",limit_size," + str(
                limit_size) + ",conv_type," +
            conv_type.name + ",mem_test," + str(
                mem_test) + ",num_epochs," + str(
                num_epochs) + ",is_debug," + str(
                is_debug) + ",optimizer_type," + optimizer_type.name +
            ",is_data_augmentation," + str(
                is_data_augmentation) + ",device," + str(device) + "\n")

    if mem_test:
        main_test()
    else:
        main()

    run_keras()
