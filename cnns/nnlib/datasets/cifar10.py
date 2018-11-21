from torchvision import transforms
import torchvision
import torch
from cnns.nnlib.datasets.transformations.dtype_transformation import DtypeTransformation
from cnns.nnlib.datasets.transformations.flat_transformation import FlatTransformation
from cnns.nnlib.utils.general_utils import DebugMode
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import NetworkType

def get_transform_train(dtype=torch.float32, signal_dimension=2):
    transformations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    transformations.append(DtypeTransformation(dtype=dtype))
    transform_train = transforms.Compose(transformations)
    return transform_train


def get_transform_test(dtype=torch.float32, signal_dimension=2):
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    transformations.append(DtypeTransformation(dtype))
    transform_test = transforms.Compose(transformations)
    return transform_test

def get_cifar10(args):
    """
    Get the MNIST dataset.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :return: main train and test loaders, as well as other params, such as
    number of classes.
    """
    sample_count = args.sample_count_limit
    is_debug = True if DebugMode[args.is_debug] is DebugMode.TRUE else False
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    num_workers = args.workers
    pin_memory = False
    if MemoryType[args.memory_type] is MemoryType.PINNED:
        pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    num_classes = 10
    width = 32 * 32
    # The size of the flat vector after the conv layers in LeNet.
    flat_size = 500
    batch_size = args.min_batch_size
    in_channels = 3  # number of channels in the input data
    out_channels = None
    signal_dimension = 1
    if NetworkType[args.network_type] is NetworkType.LE_NET:
        out_channels = [10, 20]
        signal_dimension = 2
    if NetworkType[args.network_type] is NetworkType.ResNet18:
        signal_dimension = 2
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True,
                                                 transform=get_transform_train(
                                                     dtype=torch.float,
                                                     signal_dimension=signal_dimension))
    if sample_count > 0:
        train_dataset.train_data = train_dataset.train_data[:sample_count]
        train_dataset.train_labels = train_dataset.train_labels[
                                     :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True,
                                                transform=get_transform_test(
                                                    dtype=torch.float,
                                                    signal_dimension=signal_dimension))
    if sample_count > 0:
        test_dataset.test_data = test_dataset.test_data[:sample_count]
        test_dataset.test_labels = test_dataset.test_labels[:sample_count]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)

    return train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels