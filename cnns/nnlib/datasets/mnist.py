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
    use_cuda = args.use_cuda
    num_workers = args.workers
    pin_memory = False
    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    args.num_classes = 10
    args.flat_size = 320  # the size of the flat vector after the conv layers in LeNet
    args.width = 28 * 28
    args.in_channels = 1  # number of channels in the input data
    args.out_channels = [10, 20]
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
    if sample_count > 0:
        train_dataset.train_data = train_dataset.train_data[:sample_count]
        train_dataset.train_labels = train_dataset.train_labels[
                                     :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.min_batch_size,
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
    if sample_count > 0:
        test_dataset.test_data = test_dataset.test_data[:sample_count]
        test_dataset.test_labels = test_dataset.test_labels[:sample_count]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              **kwargs)

    return train_loader, test_loader
