from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.mnist.mnist import get_mnist


def get_data(args):
    if args.dataset == "imagenet":
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet(
            args)
        limit = 50000
    elif args.dataset == "cifar10":
        train_loader, test_loader, train_dataset, test_dataset = get_cifar(
            args, args.dataset)
        limit = 10000
    elif args.dataset == "mnist":
        train_loader, test_loader, train_dataset, test_dataset = get_mnist(
            args)
        limit = 10000
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")
    return train_loader, test_loader, train_dataset, test_dataset, limit