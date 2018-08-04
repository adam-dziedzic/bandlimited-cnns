# -*- coding: utf-8 -*-
"""
Find cliffs in the execution of neural networks.
Go to the level of C++ and cuda.
Find for what input size, the memory size is not sufficient.
Run a single forward pass and a subsequent backward pass.

Define neural network, compute loss and make updates to the weights of the
network.


We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
"""

import sys
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import OptimizerType

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


def get_log_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


class LoadCifar10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LoadCifar10, self).__init__(
            root=root, train=train, transform=transform,
            target_transform=target_transform, download=download)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def define_net(input_size=32, batch_size=64, num_classes=10):
    """
    Define your model: a deep neural network.

    :return: the model architecture
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.input_channel = 3
            self.size = input_size
            self.conv1_filter_size = 5
            self.conv1_channels = 6
            self.conv1 = nn.Conv2d(self.input_channel, self.conv1_channels,
                                   self.conv1_filter_size)
            self.size = self.size - self.conv1_filter_size + 1
            self.pool = nn.MaxPool2d(2, 2)
            self.size = self.size // 2
            self.conv2_filter_size = 5
            self.conv2_channels = 16
            self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels,
                                   self.conv2_filter_size)
            self.size = self.size - self.conv2_filter_size + 1
            self.size = self.size // 2
            # print("self.size in net: ", self.size)
            self.fc1 = nn.Linear(self.conv2_channels * self.size * self.size,
                                 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            # print("x shape after conv1 and pool: ", x.shape)
            x = self.pool(F.relu(self.conv2(x)))
            # print("x shape after conv2 and pool: ", x.shape)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # from torchvision.models import AlexNet
    # print("input size: ", input_size)
    # 1st conv2d Alex Net
    # from pytorch_tutorials.memory_net_alex_1_conv2d import AlexNet
    # from pytorch_tutorials.memory_net_alex_2_conv2d import AlexNet
    # from cnns.pytorch_tutorials.memory_net_alex_fc import AlexNet
    # net = AlexNet(num_classes=num_classes, input_size=input_size)
    # from torchvision.models import DenseNet
    # from cnns.pytorch_tutorials.memory_net_densenet import DenseNet
    # net = DenseNet()
    from cnns.pytorch_tutorials.memory_net_densenet import densenet121
    net = densenet121(conv_type=conv_type)
    # net = Net()
    # from torchvision.models import resnet152
    # from cnns.pytorch_tutorials.memory_net_resnet import resnet152
    # net = resnet152()
    print("net used: ", net.__class__.__name__)
    return net


class MeasureSizePIL(object):
    """Measure the size of the image."""

    def __init__(self):
        super(MeasureSizePIL, self).__init__()
        self.counter = 0

    def __call__(self, img):
        """
        Rescale the channels.

        :param image: the input image
        :return: image size in bytes
        """
        from io import BytesIO
        img_file = BytesIO()
        img.save(img_file, 'png')
        img_file_size_png = img_file.tell()
        img_file = BytesIO()
        img.save(img_file, 'jpeg')
        img_file_size_jpeg = img_file.tell()
        print("img_file_size png: ", img_file_size_png)
        print("img_file_size jpeg: ", img_file_size_jpeg)
        print("img type: ", type(img))
        print("img size in memory in bytes: ", sys.getsizeof(img.tobytes()))
        # print("img in memory: ", img.tobytes())
        print("img mode: ", img.mode)
        print("img size w=%d, h=%d", img.size)
        self.counter += 1
        print("counter: ", self.counter)
        return img


class MeasureSizeTensor(object):
    """Measure the size of the image."""

    def __init__(self):
        super(MeasureSizeTensor, self).__init__()
        self.counter = 0

    def __call__(self, img_tensor):
        """
        Rescale the channels.

        :param image: the input image
        :return: image size in bytes

        :see https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save
        """
        from io import BytesIO
        buffer = BytesIO()
        torch.save(img_tensor, buffer)
        torch_size = buffer.tell()
        print("torch object size: ", torch_size)
        print("tensor img size in bytes in memory: ", sys.getsizeof(torch_size))
        self.counter += 1
        print("counter: ", self.counter)
        return img_tensor


class ScaleChannel(object):
    """Scale the channel of the image to the required size."""

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        Rescales the input PIL.Image to the given 'size'.
        'size' - number of channels
        :param size: the required size of channels
        :param interpolation: Default: PIL.Image.BILINEAR
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Rescale the channels.

        :param image: the input image
        :return: rescaled image with the required number of channels:return:
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        print("image type: ", type(img), " (is it a numpy array?)")
        # image = image.transpose((2, 0, 1))
        # image.resize((self.size, 0), self.interpolation)
        # image = image.transpose((1, 2, 0))
        img = transforms.ToTensor()(img)
        img = img.numpy()
        img = np.swapaxes(img, 0, 2)
        # img = img.view(img.size[1], img.size[2], img.size[0])
        img = transforms.ToPILImage()(
            img),  # go back to img to do the interpolation
        img = img.resize((0, self.size), self.interpolation)
        img = transforms.ToTensor()(img)
        # img = img.view(img.size[], img.size[2], img.size[0])
        img = img.numpy()
        img = np.swapaxes(img, 0, 2)
        img = transforms.ToPILImage()(img),  # go back to the typical pipeline
        return img


class ScaleChannel2(object):
    """Scale the channel of the image to the required size."""

    def __init__(self, size):
        """
        Rescales the input PIL.Image to the given 'size'.
        'size' - number of channels
        :param size: the required size of channels
        :param interpolation: Default: PIL.Image.BILINEAR
        """
        self.size = size

    def __call__(self, img):
        """
        Rescale the channels.

        :param image: the input image
        :return: rescaled image with the required number of channels:return:
        """
        return img[1, ...].expand(self.size, -1, -1)


def data_transformations(input_size):
    """
    Data transformations, e.g., flip, normalize.

    :param input_size: size of one of the sides of a square image
    :return: a collection of train and test transformations in an array
    """
    if is_data_augmentation:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if mem_test:
            transform_train = transforms.Compose(
                [
                    # ScaleChannel(channel_size),  # this is a hack - to be able to scale the channel size
                    # transforms.Scale(input_size),
                    # scale the input image HxW to the required size
                    # MeasureSizePIL(),
                    transforms.ToTensor(),
                    # ScaleChannel2(channel_size),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.RandomHorizontalFlip()
                    # MeasureSizeTensor(),
                ])
            transform_test = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)])
        else:
            # Data transforms: https://bit.ly/2MdSdLL
            transform_train = transforms.Compose([
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:
        transform_train = transforms.Compose(
            [transforms.ToTensor()])
        transform_test = transforms.Compose(
            [transforms.ToTensor()])
    return transform_train, transform_test


def load_data_CIFAR10(input_size=32, batch_size=64, num_workers=0,
                      channel_size=3):
    """
    Loading and normalizing CIFAR10
    """

    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    root = "./data"
    shuffle = True
    download = True

    transform_train, transform_test = data_transformations(
        input_size=input_size) if is_data_augmentation else [], []

    trainset = LoadCifar10(root=root, train=True, download=download,
                           transform=transform_train)
    print("The size of the train dataset: ", len(trainset.train_data))
    if limit_size > 0:
        print("Limit the train input size for debug purposes:")
        trainset.train_data = trainset.train_data[:limit_size]
        trainset.train_labels = trainset.train_labels[:limit_size]
        print("The size of the train dataset after limitation: ",
              len(trainset.train_data))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    testset = LoadCifar10(root=root, train=False, download=download,
                          transform=transform_test)
    print("The size of the test dataset: ", len(testset.test_data))
    if limit_size > 0:
        print("Limit the test input size for debug purposes:")
        testset.test_data = testset.test_data[:limit_size]
        testset.test_labels = testset.test_labels[:limit_size]
        print("The size of the test dataset after limitation: ",
              len(testset.test_data))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    return trainloader, testloader, classes


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
                # shrink to 1 layer and then expand to the required number of channels
                # inputs = torch.tensor(inputs[:, 0:1, ...].expand(-1, net.input_channel, -1, -1))
                # generate the random data of required image size and the number of channels
                inputs = torch.randn(batch_size, net.input_channel,
                                     net.img_size_to_features,
                                     net.img_size_to_features).to(device)

            start_total = time.time()

            # zero the parameter gradients before computing gradient for the new batch
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
                    '[%d, %5d],forward time,%f,backward_time,%f,optimizer_time,%f,total_time,%f,loss,%.3f,'
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
    Test the network on the testing set.

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
    if optimizer_type is OptimizerType.ADAM:
        optimizer = optim.Adam(params=net.parameters(),
                               weight_decay=weight_decay)
    elif optimizer_type in [OptimizerType.SGD or OptimizerType.MOMENTUM]:
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr,
                                    momentum=momentum, nesterov=True,
                                    weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[
                                                         0.5 * num_epochs,
                                                         0.75 * num_epochs],
                                                     gamma=0.1)

    return optimizer, scheduler


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

    optimizer, scheduler = get_optimizer(net, optimizer_type)

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
    # if we are on a CUDA machine, then this should print a CUDA device (otherwise it prints cpu):
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

    log_file = get_log_time() + ".log"
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
