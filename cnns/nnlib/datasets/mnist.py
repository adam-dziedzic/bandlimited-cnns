import torchvision
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import DebugMode
from cnns.nnlib.utils.general_utils import MemoryType


def get_mnist(args):
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
    flat_size = 320  # the size of the flat vector after the conv layers in LeNet
    width = 28 * 28
    in_channels = 1  # number of channels in the input data
    out_channels = [10, 20]
    batch_size = args.min_batch_size
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,),
                    (0.3081,))
            ])
    )
    if is_debug and sample_count > 0:
        train_dataset.train_data = train_dataset.train_data[:sample_count]
        train_dataset.train_labels = train_dataset.train_labels[
                                     :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,),
                    (0.3081,))
            ])
    )
    if is_debug and sample_count > 0:
        test_dataset.test_data = test_dataset.test_data[:sample_count]
        test_dataset.test_labels = test_dataset.test_labels[:sample_count]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)

    return train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels
