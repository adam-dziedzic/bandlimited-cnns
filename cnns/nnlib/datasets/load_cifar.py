import torch
from torch import tensor
import torchvision
import torchvision.transforms as transforms
from cnns.nnlib.datasets.cifar import cifar_mean, cifar_std


def denormalize(tensor, mean, std):
    return tensor.mul_(std).add_(mean)


def normalize(tensor, mean, std):
    return tensor.sub_(mean).div_(std)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(cifar_mean, cifar_std)])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    img.mul_(tensor(cifar_std).view(3, 1, 1)).add_(
        tensor(cifar_mean).view(3, 1, 1))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

torch.set_printoptions(profile="full")
print(images[0], labels[0])
torch.set_printoptions(profile="default")

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
