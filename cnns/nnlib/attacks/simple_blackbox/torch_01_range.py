"""
Transform between torch (from torchvision) range of pixel values and zero-one
pixel values.
"""
import torch


class Ranger():
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.mnist_mean = torch.tensor([0.1307], device=device).view((1, 1, 1))
        self.mnist_std = torch.tensor([0.3081], device=device).view((1, 1, 1))

        self.cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view((3, 1, 1))
        self.cifar_std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view((3, 1, 1))

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((3, 1, 1))
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view((3,1, 1))


    def to_01_mnist(self, x):
        return x * self.mnist_std.to(x.device) + self.mnist_mean.to(x.device)

    def to_01_cifar(self, x):
        return x * self.cifar_std.to(x.device) + self.cifar_mean.to(x.device)

    def to_01_imagenet(self, x):
        return x * self.imagenet_std.to(x.device) + self.imagenet_mean.to(x.device)

    def to_01(self, x, dataset='imagenet'):
        if dataset == 'mnist':
            return self.to_01_mnist(x)
        elif dataset == 'cifar':
            return self.to_01_cifar(x)
        elif dataset == 'imagenet':
            return self.to_01_imagenet(x)
        else:
            raise Exception(f'Unknown dataset: {dataset}')

    def to_torch_mnist(self, x):
        return (x - self.mnist_mean.to(x.device)) / self.mnist_std.to(x.device)

    def to_torch_cifar(self, x):
        return (x - self.cifar_mean.to(x.device)) / self.cifar_std.to(x.device)

    def to_torch_imagenet(self, x):
        return (x - self.imagenet_mean.to(x.device)) / self.imagenet_std.to(x.device)

    def to_torch(self, x, dataset='imagenet'):
        if dataset == 'mnist':
            return self.to_torch_mnist(x)
        elif dataset == 'cifar':
            return self.to_torch_cifar(x)
        elif dataset == 'imagenet':
            return self.to_torch_imagenet(x)
        else:
            raise Exception(f'Unknown dataset: {dataset}')
