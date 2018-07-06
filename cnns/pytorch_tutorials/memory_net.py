# -*- coding: utf-8 -*-
"""
Find cliffs in the execution of neural networks.
Go to the level of C++ and cuda.
Find for what input size the memory size is not sufficient.
Run a single forward pass and a subsequent backward pass.

Define neural network, compute loss and make updates to the weights of the network.


We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using ``torchvision``
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# switch backend to be able to save the graphic files on the servers
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--iterations", default=1, type=int, help="number of iterations for the training")
parser.add_argument("-i", "--initbatchsize", default=64, type=int,
                    help="the initial size of the batch (number of data points for a single forward and batch passes")
parser.add_argument("-m", "--maxbatchsize", default=64, type=int,
                    help="the max size of the batch (number of data points for a single forward and batch passes")
parser.add_argument("-s", "--startsize", default=64, type=int, help="the start size of the input")
parser.add_argument("-e", "--endsize", default=1000000, type=int, help="the end size of the input")
parser.add_argument("-w", "--workers", default=0, type=int,
                    help="number of workers to fetch data for pytorch data loader, 0 means that the data will be "
                         "loaded in the main process")

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)


def get_log_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


def define_net(input_size=32, batch_size=64):
    """
    Define your model: a deep neural network.

    :return: the model architecture
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.size = input_size
            self.conv1_filter_size = 5
            self.conv1_channels = 6
            self.conv1 = nn.Conv2d(3, self.conv1_channels, self.conv1_filter_size)
            self.size = self.size - self.conv1_filter_size + 1
            self.pool = nn.MaxPool2d(2, 2)
            self.size = self.size // 2
            self.conv2_filter_size = 5
            self.conv2_channels = 16
            self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels, self.conv2_filter_size)
            self.size = self.size - self.conv2_filter_size + 1
            self.size = self.size // 2
            # print("self.size in net: ", self.size)
            self.fc1 = nn.Linear(self.conv2_channels * self.size * self.size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

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
    from pytorch_tutorials.memory_net_alex import AlexNet
    # print("input size: ", input_size)
    net = AlexNet(input_size=input_size)
    # net = Net()
    return net


def load_data(input_size=32, batch_size=64, num_workers=0):
    """
    Loading and normalizing CIFAR10
    """

    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    root = "./data"
    shuffle = False
    download = False

    transform = transforms.Compose(
        [transforms.Scale(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=download, transform=transform)
    # print("The size of the train dataset: ", len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=download, transform=transform)
    # print("The size of the test dataset: ", len(trainset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def train_network(net, trainloader, optimizer, criterion, device=torch.device("cpu")):
    """
    Train the network

    Loop over the data iterator, and feed the inputs to the network and optimize.
    """
    # optimization
    iter_number_print = iter_number_total
    forward_time = []
    backward_time = []
    optimizer_time = []
    aggregator = lambda array: np.median(array, overwrite_input=False)

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            # get the inputs
            inputs, labels = data
            # move them to CUDA (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
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
            if i % iter_number_print == iter_number_print - 1:  # print every 1 mini-batch
                print('[%d, %5d],forward time,%f,backward_time,%f,optimizer_time,%f,loss,%.3f,input_size,%d' %
                      (
                          epoch + 1, i + 1, aggregator(forward_time), aggregator(backward_time),
                          aggregator(optimizer_time),
                          running_loss / iter_number_print,
                          inputs.shape[-1]))
                running_loss = 0.0
            if i + 1 == iter_number_total:
                break

    return aggregator(forward_time), aggregator(backward_time), aggregator(optimizer_time)


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

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Let us see what the neural network thinks these examples above are:

    outputs = net(images)

    ########################################################################
    # The outputs are energies for the 10 classes.
    # Higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

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
    print('Accuracy of the network on the 10000 test images: %d %%' % (final_accuracy))

    # what are the classes that performed well, and the classes that did not perform well:
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
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    return final_accuracy


def main():
    input_size = 64
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if we are on a CUDA machine, then this should print a CUDA device (otherwise it prints cpu):
    print("Currently used device: ", device)
    net = define_net(input_size=input_size, batch_size=batch_size)
    net.to(device)

    trainloader, testloader, classes = load_data(input_size=input_size, batch_size=batch_size)

    # Define a Loss function and optimizer
    # Use a Classification Cross-Entropy loss and SGD with momentum.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_network(net=net, optimizer=optimizer, criterion=criterion, trainloader=trainloader, device=device)
    print('Finished Training')

    test_network(net=net, testloader=testloader, classes=classes, device=device)


def plot_figure(batch_forward_times, batch_backward_times, batch_input_sizes, batch_sizes, iter_number_total):
    fig, ax = plt.subplots()
    for batch_index, batch_size in enumerate(batch_sizes):
        input_sizes = batch_input_sizes[batch_index]
        forward_times = batch_forward_times[batch_index]
        backward_times = batch_backward_times[batch_index]

        forward_label = "forward pass for batch size " + str(batch_size)
        ax.plot(input_sizes, forward_times, label=forward_label)
        backward_label = "backward pass for batch size " + str(batch_size)
        ax.plot(input_sizes, backward_times, label=backward_label)

    ax.legend()
    input_sizes_example = batch_input_sizes[0]
    # fig.set_xticklables(input_sizes_example)

    ax.set_yscale('log')
    ax.set_xscale('log')

    # plt.xticks(input_sizes_example)

    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title('Compare execution time of forward and backward passes')
    # plt.suptitle('iteration number per training: ' + str(iter_number_total), fontsize=12)
    plt.xlabel('Input size (size x size image)')
    plt.ylabel('Execution time (sec)')
    file_name = "graphs/mem-forward-backward-" + current_file_name + "-" + get_log_time() + "-iterations-" + str(
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if we are on a CUDA machine, then this should print a CUDA device (otherwise it prints cpu):
    print("Currently used device: ", device)
    batch_sizes = []
    batch_forward_times = []
    batch_backward_times = []
    batch_optimizer_times = []
    batch_input_sizes = []

    batch_size = init_batch_size
    while batch_size <= max_batch_size:
        print("batch size: ", batch_size)
        forward_times = []
        backward_times = []
        optimizer_times = []
        input_sizes = []

        # try:
        input_size = start_size
        while input_size <= end_size:
            # print("input size: ", input_size)
            net = define_net(input_size=input_size, batch_size=batch_size)
            net.to(device)

            trainloader, testloader, classes = load_data(input_size=input_size, batch_size=batch_size,
                                                         num_workers=num_workers)

            # Define a Loss function and optimizer
            # Use a Classification Cross-Entropy loss and SGD with momentum.
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            forward_time, backward_time, optimizer_time = train_network(net=net, optimizer=optimizer,
                                                                        criterion=criterion,
                                                                        trainloader=trainloader, device=device)

            forward_times.append(forward_time)
            backward_times.append(backward_time)
            optimizer_times.append(optimizer_time)

            input_sizes.append(input_size)
            input_size *= 2

            torch.cuda.empty_cache()

        # except RuntimeError as err:
        #     print("Runtime error: " + str(err))

        batch_sizes.append(batch_size)
        batch_size *= 2

        batch_forward_times.append(forward_times)
        batch_backward_times.append(backward_times)
        batch_optimizer_times.append(optimizer_times)
        batch_input_sizes.append(input_sizes)

        print("batch_sizes=", batch_sizes)
        print("batch_input_sizes=", batch_input_sizes)
        print("batch_forward_times=", batch_forward_times)
        print("batch_backward_times=", batch_backward_times)
        print("batch_optimizer_times=", batch_optimizer_times)

    plot_figure(batch_forward_times=batch_forward_times, batch_backward_times=batch_backward_times,
                batch_input_sizes=batch_input_sizes, batch_sizes=batch_sizes, iter_number_total=iter_number_total)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    iter_number_total = args.iterations
    init_batch_size = args.initbatchsize
    max_batch_size = args.maxbatchsize
    start_size = args.startsize
    end_size = args.endsize
    num_workers = args.workers

    # main()
    main_test()
