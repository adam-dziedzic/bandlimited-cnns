from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel


def get_dev_dataset(args, train_dataset):
    """
    Get the dev set as args.dev_percent of the last rows from the train set.

    :param args: the args of the program
    :param train_dataset: the train dataset
    :param dataset_name: the name of the dataset
    :return: the dev set
    """
    dataset_name = args.dataset_name
    dev_percent = args.dev_percent
    if dev_percent <= 0 or dev_percent >= 100:
        raise Exception(f"Dev set was declared to be used but the percentage "
                        "of the train set to be used as the dev set was "
                        "mis-specified with value: {dev_percent}.")
    train_percent = 100 - dev_percent
    total_len = len(train_dataset)
    train_len = int(total_len * train_percent / 100)
    dev_len = total_len - train_len

    train_dataset.set_range(0, train_len)
    dev_dataset = UCRDataset(dataset_name, train=True,
                             transformations=transforms.Compose(
                                 [ToTensor(dtype=torch.float),
                                  AddChannel()]))
    dev_dataset.set_range(train_len, total_len)
    if len(dev_dataset) != dev_len:
        raise Exception("Error in extracting the dev set from the train set.")
    return dev_dataset


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
                                    AddChannel()]),
                               ucr_path=args.ucr_path)
    if sample_count > 0:
        train_dataset.set_length(sample_count)

    train_size = len(train_dataset)

    if args.is_dev_dataset:
        dev_dataset = get_dev_dataset(args=args, train_dataset=train_dataset)

    batch_size = args.min_batch_size
    if train_size < batch_size:
        batch_size = train_size

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        **kwargs)

    dev_loader = None
    if args.is_dev_dataset:
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset, batch_size=batch_size, shuffle=True,
            **kwargs)

    test_dataset = UCRDataset(dataset_name, train=False,
                              transformations=transforms.Compose(
                                  [ToTensor(dtype=torch.float),
                                   AddChannel()]),
                              ucr_path=args.ucr_path)
    if sample_count > 0:
        test_dataset.set_length(sample_count)

    args.num_classes = test_dataset.num_classes
    args.input_size = test_dataset.width
    # args.min_batch_size = int(min(args.input_size / 10, args.min_batch_size))
    # args.test_batch_size = int(min(args.input_size / 10, args.test_batch_size))
    args.flat_size = None
    args.out_channels = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, dev_loader
