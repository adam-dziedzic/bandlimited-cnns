import numpy as np
import os

database_path = '/TimeSeriesDatasets/'


def load_data(dirname, normalization=False, slice_ratio=1, percent_valid=0.2):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print("current path: ", dir_path)

    rng = np.random.RandomState(23455)
    train_file = dir_path + database_path + dirname + '/' + dirname + '_TRAIN'
    test_file = dir_path + database_path + dirname + '/' + dirname + '_TEST'

    # load train set
    data = np.loadtxt(train_file, dtype=np.str, delimiter=",")
    train_x = data[:, 1:].astype(np.float32)
    train_y = np.int_(data[:, 0].astype(np.float32)) - 1  # label starts from 0
    # print("shape of the train_x: ", train_x.shape)
    # print("train size before splitting: ", len(train_y))
    len_train_data = train_x.shape[1]

    # restrict slice ratio when data length is too large
    if len_train_data > 500:
        slice_ratio = slice_ratio if slice_ratio > 0.98 else 0.98

    # shuffle for splitting train set and dataset
    n = train_x.shape[0]
    ind = np.arange(n)
    rng.shuffle(ind)  # shuffle the train set

    # split train set into train set and validation set
    if percent_valid > 0:
        valid_last_index = int(percent_valid * n)
        valid_x = train_x[ind[:valid_last_index]]
        valid_y = train_y[ind[:valid_last_index]]

        ind = np.delete(ind, (range(0, int(valid_last_index))))

        train_x = train_x[ind]
        train_y = train_y[ind]

    # print("size of train set: ", len(train_y))
    # print("size of validation set: ", len(valid_y))

    train_x, train_y = slice_data(train_x, train_y, slice_ratio)
    valid_x, valid_y = slice_data(valid_x, valid_y, slice_ratio)

    # shuffle again
    n = train_x.shape[0]
    ind = np.arange(n)
    rng.shuffle(ind)  # shuffle the train set

    # load test set
    data = np.loadtxt(test_file, dtype=np.str, delimiter=",")
    test_x = data[:, 1:].astype(np.float32)
    test_y = np.int_(data[:, 0].astype(np.float32)) - 1

    test_x, test_y = slice_data(test_x, test_y, slice_ratio)
    # print("size of the test set: ", len(test_x))

    # z-normalization (not done by default - the UCR dataset is normalized already)
    if normalization:
        mean_x = train_x.mean(axis=0)
        std_x = train_x.std(axis=0)
        train_x = (train_x - mean_x) / std_x
        valid_x = (valid_x - mean_x) / std_x
        test_x = (test_x - mean_x) / std_x

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y), len_train_data, slice_ratio]


def slice_data(data_x, data_y, slice_ratio=1):
    # return the sliced dataset
    if slice_ratio == 1:
        return data_x, data_y
    n = data_x.shape[0]
    length = data_x.shape[1]
    length_sliced = int(length * slice_ratio)

    increase_num = length - length_sliced + 1  # if increase_num =5, it means one ori becomes 5 new instances.
    n_sliced = n * increase_num
    # print "*increase num", increase_num
    # print "*new length", n_sliced, "orginal len", n

    new_x = np.zeros((n_sliced, length_sliced))
    new_y = np.zeros((n_sliced))
    for i in range(n):
        for j in range(increase_num):
            new_x[i * increase_num + j, :] = data_x[i, j: j + length_sliced]
            new_y[i * increase_num + j] = np.int_(data_y[i].astype(np.float32))
    return new_x, new_y
