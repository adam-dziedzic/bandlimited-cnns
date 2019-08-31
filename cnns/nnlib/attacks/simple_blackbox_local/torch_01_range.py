"""
Transform between torch (from torchvision) range of pixel values and zero-one
pixel values.
"""
import torch


mnist_mean = torch.tensor([0.1307]).view((1, 1, 1))
mnist_std = torch.tensor([0.3081]).view((1, 1, 1))

cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view((3, 1, 1))
cifar_std = torch.tensor([0.2023, 0.1994, 0.2010]).view((3, 1, 1))

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view((3,1, 1))


def to_01_mnist(x):
    return x * mnist_std + mnist_mean

def to_01_cifar(x):
    return x * cifar_std + cifar_mean

def to_01_imagenet(x):
    return x * imagenet_std + imagenet_std

def to_01(x, dataset='imagenet'):
    if dataset == 'mnist':
        return to_01_mnist(x)
    elif dataset == 'cifar':
        return to_01_cifar(x)
    elif dataset == 'imagenet':
        return to_01_imagenet(x)
    else:
        raise Exception(f'Unknown dataset: {dataset}')

def to_torch_mnist(x):
    return (x - mnist_mean) / mnist_std

def to_torch_cifar(x):
    return (x - cifar_mean) / cifar_std

def to_torch_imagenet(x):
    return (x - imagenet_mean) / imagenet_std

def to_torch(x, dataset='imagenet'):
    if dataset == 'mnist':
        return to_torch_mnist(x)
    elif dataset == 'cifar':
        return to_torch_cifar(x)
    elif dataset == 'imagenet':
        return to_torch_imagenet(x)
    else:
        raise Exception(f'Unknown dataset: {dataset}')
