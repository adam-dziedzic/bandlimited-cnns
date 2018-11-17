from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import DebugMode
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel

def get_ucr(args, dataset_name):
    """
    Get a dataset from the UCR archive.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :param dataset_name: the name of a dataset from the ucr archive
    :return: the access handlers to the dataset
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
    in_channels = 1  # number of channels in the input data
    train_dataset = UCRDataset(dataset_name, train=True,
                               transformations=transforms.Compose(
                                   [ToTensor(dtype=torch.float),
                                    AddChannel()]))
    if is_debug and sample_count > 0:
        train_dataset.set_length(sample_count)
    train_size = len(train_dataset)
    batch_size = args.min_batch_size
    if train_size < batch_size:
        batch_size = train_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        **kwargs)

    test_dataset = UCRDataset(dataset_name, train=False,
                              transformations=transforms.Compose(
                                  [ToTensor(dtype=torch.float),
                                   AddChannel()]))
    if is_debug and sample_count > 0:
        test_dataset.set_length(sample_count)
    num_classes = test_dataset.num_classes
    width = test_dataset.width
    # We don't use the 1D data with LeNet
    flat_size = None
    out_channels = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, num_classes, flat_size, width, in_channels, out_channels