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


class RemyDataset(Dataset):
    def __init__(
            self,
            train=True,
            data_path=None):
        """
        :param dataset_name: the name of the dataset to fetch from file on disk.
        :param transformations: pytorch transforms for transforms and tensor
        conversion.
        :param data_path: the path to the ucr dataset.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if data_path is None:
            data_path = os.path.join(dir_path, os.pardir, os.pardir,
                                     "TimeSeriesDatasets")
        else:
            data_path = os.path.join(dir_path, data_path)
        if train is True:
            suffix = "_train"
        else:
            suffix = "_test"
        csv_path = data_path + '/' + 'remy_data' + suffix
        self.data_all = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data_all.iloc[:, 0], dtype=np.int)
        self.length = len(self.labels)
        self.num_classes = len(np.unique(self.labels))
        self.labels = self.__transform_labels(labels=self.labels,
                                              num_classes=self.num_classes)
        self.data = np.asarray(self.data_all.iloc[:, 1:], dtype=np.float)

        # randomize the data
        randomized_indices = np.random.choice(
            self.length, self.length, replace=False)
        self.data = self.data[randomized_indices, ...]
        self.labels = self.labels[randomized_indices]

        self.width = len(self.data[0, :])
        self.dtype = torch.float
        self.data = torch.tensor(self.data, device=torch.device("cpu"),
                                 dtype=self.dtype)
        # add the dimension for the channel
        self.data = torch.unsqueeze(self.data, dim=1)

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
        # input = np.asarray(self.data.iloc[index][1:])
        input = self.data[index]
        # Transform time-series input to tensor.
        # if self.transformations is not None:
        #     input = self.transformations(input)
        # Return the time-series and the label.
        return input, label

    def __len__(self):
        # self.data.index - The index(row labels) of the DataFrame.
        # length = len(self.data.index)
        length = len(self.data)
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

    def set_range(self, start, stop):
        """
        :param start: the start row
        :param stop: the last row (exclusive) of the dataset

        :return: the dataset with the specified range.
        """
        assert len(self.data) == len(self.labels)
        self.data = self.data[start:stop]
        self.labels = self.labels[start:stop]


if __name__ == "__main__":
    train_dataset = RemyDataset(train=True,
                                transformations=transforms.Compose(
                                    [ToTensor(dtype=torch.float),
                                     AddChannel()]))
    print('first data item: ', train_dataset[0])
    print("length of the train dataset: ", len(train_dataset))
