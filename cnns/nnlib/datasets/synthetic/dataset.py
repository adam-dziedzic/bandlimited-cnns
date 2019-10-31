from torch.utils.data import Dataset
import numpy as np
from random import randint


class SyntheticDataset(Dataset):
    """Synthetic dataset that has the same number in the whole tensor
    (the number is the same as the label of the class).
    This is purely data of rank 1."""

    def __init__(
            self, train=True, train_len=60000, test_len=60000,
            num_classes=10, precompute_set=True,
            width=28, height=28, transform=None):
        if train:
            self.length = train_len
        else:
            self.length = test_len
        self.width = width
        self.height = height
        self.index = 0
        self.num_classes = num_classes
        self.precompute_set = precompute_set
        self.transform = transform

        if self.precompute_set:
            labels = []
            data = []
            class_len = self.length // num_classes
            assert self.length % num_classes == 0
            for class_nr in range(num_classes):
                labels.append(class_nr * np.ones(class_len, dtype=np.long))
                data.append(class_nr * np.ones(
                    (class_len, 1, self.height, self.width),
                    dtype=np.float32))
            self.labels = np.concatenate(labels)
            self.data = np.concatenate(data, axis=0)

            # randomize the data
            randomized_indices = np.random.choice(
                self.length, self.length, replace=False)
            self.data = self.data[randomized_indices, ...]
            self.labels = self.labels[randomized_indices]

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
        assert index >= 0 and index < self.length
        if self.precompute_set:
            label = self.labels[index]
            input = self.data[index]
        else:
            label = randint(0, self.num_classes - 1)
            # First dimension of size 1 is for the single channel.
            input = label * np.ones((1, self.height, self.width),
                                    dtype=np.float32)
        if self.transform is not None:
            input = self.transform(input)
        return input, label

    def __len__(self):
        return self.length

    def set_length(self, length):
        """
        :param length: The length of the datasets (a subset of data points),
        first length samples.
        """
        self.length = length

    def set_range(self, start, stop):
        """
        :param start: the start row
        :param stop: the last row (exclusive) of the dataset

        :return: the dataset with the specified range.
        """
        self.index = 0
        self.length = stop - start
        return self


if __name__ == "__main__":
    train_set = SyntheticDataset(train=True)
    print('data item 0: ', train_set[0])
    labels = []
    for index, (data, label) in enumerate(train_set):
        labels.append(label)
        if index == train_set.length - 1:
            break

    counter = {}
    for label in labels:
        if counter.get(label):
            counter[label] += 1
        else:
            counter[label] = 1
    print('counter: ', counter)
