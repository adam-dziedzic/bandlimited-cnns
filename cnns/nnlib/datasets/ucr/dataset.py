import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np


class ToTensor(object):
    """Transform the numpy array to a tensor."""

    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, input):
        """
        :param input: numpy array.
        :return: PyTorch's tensor.
        """
        # Transform data on the cpu.
        return torch.tensor(input, device=torch.device("cpu"),
                            dtype=self.dtype)


class AddChannel(object):
    """Add channel dimension to the input time series."""

    def __call__(self, input):
        """
        Rescale the channels.

        :param image: the input image
        :return: rescaled image with the required number of channels:return:
        """
        # We receive only a single array of values as input, so have to add the
        # channel as the zero-th dimension.
        return torch.unsqueeze(input, dim=0)


class UCRDataset(Dataset):
    """One of the time-series datasets from the UCR archive."""

    def __init__(
            self, dataset_name, transformations=transforms.Compose(
                [ToTensor(), AddChannel()]), train=True,
            ucr_path=None):
        """
        :param dataset_name: the name of the dataset to fetch from file on disk.
        :param transformations: pytorch transforms for transforms and tensor
        conversion.
        :param ucr_path: the path to the ucr dataset.
        """
        if ucr_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            ucr_path = os.path.join(dir_path, os.pardir, os.pardir,
                                    "TimeSeriesDatasets")
        if train is True:
            suffix = "_TRAIN"
        else:
            suffix = "_TEST"
        csv_path = os.path.join(ucr_path, dataset_name, dataset_name + suffix)
        self.data = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.num_classes = len(np.unique(self.labels))
        self.labels = self.__transform_labels(labels=self.labels,
                                              num_classes=self.num_classes)
        self.width = len(np.asarray(self.data.iloc[0, 1:]))
        self.transformations = transformations

    @staticmethod
    def __transform_labels(labels, num_classes):
        """
        Start class numbering from 0, and provide them in range from 0 to
        self.num_classes - 1.

        Example:
        y_train = np.array([-1, 2, 3, 3, -1, 2])
        nb_classes = 3
        ((y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)).astype(int)
        Out[45]: array([0, 1, 2, 2, 0, 1])

        >>> labels = __transofrm_labels(labels = np.array([-1, 2, 3, 3, -1, 2]),
        ... num_classes=3)
        >>> np.testing.assert_arrays_equal(x=labels,
        ... y=np.array([0, 1, 2, 2, 0, 1]))

        :param labels: labels.
        :param num_classes: number of classes.

        :return: transformed labels.
        """
        # The nll (negative log likelihood) loss requires target labels to be of
        # type Long:
        # https://discuss.pytorch.org/t/expected-object-of-type-variable-torch-longtensor-but-found-type/11833/3?u=adam_dziedzic
        return ((labels - labels.min()) / (labels.max() - labels.min()) * (
                num_classes - 1)).astype(np.int64)

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, val):
        self.__width = val

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, val):
        self.__num_classes = val

    def __getitem__(self, index):
        label = self.labels[index]
        # Take the row index and all values starting from the second column.
        input = np.asarray(self.data.iloc[index][1:])
        # Transform time-series input to tensor.
        if self.transformations is not None:
            input = self.transformations(input)
        # Return the time-series and the label.
        return input, label

    def __len__(self):
        # self.data.index - The index(row labels) of the DataFrame.
        length = len(self.data.index)
        assert length == len(self.labels)
        return length

    def set_length(self, length):
        """
        :param length: The lenght of the datasets (a subset of data points),
        first length samples.
        """
        assert len(self.data) == len(self.labels)
        self.data = self.data[:length]
        self.labels = self.labels[:length]