from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel

def get_ucr(args):
    """
    Get a dataset from the UCR archive.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :param dataset_name: the name of a dataset from the ucr archive
    :return: the access handlers to the dataset
    """
    dataset_name = args.dataset_name
    sample_count = args.sample_count_limit
    use_cuda = args.use_cuda
    num_workers = args.workers

    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    else:
        pin_memory = False
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    args.in_channels = 1  # number of channels in the input data
    train_dataset = UCRDataset(dataset_name, train=True,
                               transformations=transforms.Compose(
                                   [ToTensor(dtype=torch.float),
                                    AddChannel()]))
    if sample_count > 0:
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
    if sample_count > 0:
        test_dataset.set_length(sample_count)

    args.num_classes = test_dataset.num_classes
    args.input_size = test_dataset.width
    args.flat_size = None
    args.out_channels = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader