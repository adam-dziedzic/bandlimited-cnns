from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from torchvision import transforms
import torch
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel


def get_dev_dataset(train_dataset, dataset_name, dev_percent):
    """
    Get the dev set as args.dev_percent of the last rows from the train set.

    :param train_dataset: the train dataset
    :param dataset_name: the name of the dataset
    :param dev_percent: the percentage of the train set to be used as the
    dev set
    :return: the dev set
    """
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


def get_ucr(dataset_name, data_path, sample_count=0, use_cuda=True,
            num_workers=4, dev_percent=0, batch_size=32):
    """
    Get a dataset from the UCR archive.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :param dataset_name: the name of a dataset from the ucr archive
    :return: the access handlers to the dataset
    """

    if use_cuda and torch.cuda.is_available():
        pin_memory = True
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    train_dataset = UCRDataset(dataset_name, train=True,
                               transformations=transforms.Compose(
                                   [ToTensor(dtype=torch.float),
                                    AddChannel()]),
                               ucr_path=data_path)
    if sample_count > 0:
        train_dataset.set_length(sample_count)

    if dev_percent > 0:
        dev_dataset = get_dev_dataset(train_dataset=train_dataset,
                                      dataset_name=dataset_name + "_dev",
                                      dev_percent=dev_percent)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        **kwargs)

    dev_loader = None
    if dev_percent > 0:
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset, batch_size=batch_size, shuffle=True,
            **kwargs)

    test_dataset = UCRDataset(dataset_name, train=False,
                              transformations=transforms.Compose(
                                  [ToTensor(dtype=torch.float),
                                   AddChannel()]),
                              ucr_path=data_path)
    if sample_count > 0:
        test_dataset.set_length(sample_count)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, dev_loader
