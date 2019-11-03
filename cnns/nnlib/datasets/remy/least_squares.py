import numpy as np
from scipy.stats import describe
import os
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import qr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/
# optimal-colors-for-graphs/
MY_BLUE = (57, 106, 177)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_RED = (204, 37, 41)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)
MY_VIOLET = (107, 76, 154)
MY_BROWN = (146, 36, 40)
MY_OWN = (25, 150, 10)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


def plot(nr_samples, error_rates,
         title='error rate vs. # of train samples'):
    plt.plot(nr_samples, error_rates, color=get_color(MY_RED))
    plt.title(title)
    plt.xlabel('# of train samples')
    plt.ylabel('Error rate')
    file_name = title.replace(': ', '_').replace(' ', '_')
    file_name = file_name.replace(', ', '_').replace('.', '-')
    # plt.savefig("./iris_" + file_name + "2.pdf")
    plt.savefig("./iris.pdf")
    plt.clf()
    plt.close()


def find_w(X, y):
    X_t = X.transpose()
    X_t_X = np.matmul(X_t, X)
    X_t_X_inv = np.linalg.inv(X_t_X)
    X_t_X_inv_X_t = np.matmul(X_t_X_inv, X_t)
    w_hat = np.matmul(X_t_X_inv_X_t, y)
    return w_hat


def take_n_samples_each_clas(X, Y, nr_class, nr_samples_each_class):
    n, _ = X.shape
    n_class = n // nr_class
    x = []
    y = []
    start_index = 0
    end_index = n_class
    # We need to extract samples for each class separately
    # ensure that there are the same number of samples for
    # each class in the train and the validation sets.
    for i in range(nr_class):
        x.append(X[start_index:end_index, ...])
        y.append(Y[start_index:end_index])
        start_index += n_class
        end_index += n_class
        # Randomize the samples within this class.
        # We could also do it after the extraction
        # of the validation set.
        randomized_indices = np.random.choice(
            n_class, nr_samples_each_class, replace=False)
        x[i] = x[i][randomized_indices]
        y[i] = y[i][randomized_indices]

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return x, y


def cross_validate(X, Y, cv_count=6, nr_class=2, repeat=3,
                   train_limit=None, is_affine=True):
    """
    Cross-validate the model.

    :param X: the input matrix of features
    We expect that the samples for each class are of
    the same number and arranged in the continuous way in
    the input dataset.
    :param Y: the input vector of correct predictions
    :param cv_count: cross validation count
    how many subsets of the data we want, where
    one of the subsets is the validation set
    and the remaining subsets create constitute
    the train set. We have cv_count iterations,
    where each of the cv_count subsets is
    validation set in one of the iterations.
    :param nr_class: number of classes in the dataset
    :param repeat: how many times to repeat the process
    :param is_affine: add the column with all 1's (ones)
    :return: the average error rate across all repetitions
    and cross-validations within the repetitions.
    """
    n, _ = X.shape
    n_class = n // nr_class
    # number of samples per class
    assert n_class % cv_count == 0
    # length of the validated set from a single class
    cv_len = n_class // cv_count
    all_err = []

    for _ in range(repeat):
        x = []
        y = []
        start_index = 0
        end_index = n_class
        # We need to extract samples for each class separately
        # ensure that there are the same number of samples for
        # each class in the train and the validation sets.
        for i in range(nr_class):
            x.append(X[start_index:end_index, ...])
            y.append(Y[start_index:end_index])
            start_index += n_class
            end_index += n_class
            # Randomize the samples within this class.
            # We could also do it after the extraction
            # of the validation set.
            randomized_indices = np.random.choice(
                n_class, n_class, replace=False)
            x[i] = x[i][randomized_indices, ...]
            y[i] = y[i][randomized_indices]

        # Cross-vlidate the model cv_count times.
        for i in range(cv_count):
            bottom_index = i * cv_len
            top_index = (i + 1) * cv_len
            bottom_x = []
            top_x = []
            bottom_y = []
            top_y = []

            for j in range(nr_class):
                bottom_x.append(x[j][:bottom_index, :])
                top_x.append(x[j][top_index:, :])
                bottom_y.append(y[j][:bottom_index])
                top_y.append(y[j][top_index:])

            bottom_x = np.concatenate(bottom_x, axis=0)
            top_x = np.concatenate(top_x, axis=0)
            bottom_y = np.concatenate(bottom_y, axis=0)
            top_y = np.concatenate(top_y, axis=0)

            if i == 0:
                x_train = top_x
                y_train = top_y
            elif i == cv_count - 1:
                x_train = bottom_x
                y_train = bottom_y
            else:
                x_train = np.concatenate((bottom_x, top_x), axis=0)
                y_train = np.concatenate((bottom_y, top_y), axis=0)

            if train_limit:
                x_train = x_train[:train_limit, :]
                y_train = y_train[:train_limit]

            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)

            # normalize train
            x_train = (x_train - mean) / std

            # make it affine
            if is_affine:
                ones = np.ones(x_train.shape[0])
                x_train = np.concatenate((ones[:, np.newaxis], x_train), axis=1)

            x_test = []
            y_test = []
            for j in range(nr_class):
                x_test.append(x[j][bottom_index:top_index, :])
                y_test.append(y[j][bottom_index:top_index])

            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            x_test = (x_test - mean) / std

            # make it affine
            if is_affine:
                ones = np.ones(x_test.shape[0])
                x_test = np.concatenate((ones[:, np.newaxis], x_test), axis=1)


            w = find_w(x_train, y_train)
            y_hat = np.rint(np.matmul(x_test, w))
            diff = np.sum(y_hat != y_test)
            err = diff / (cv_len * nr_class)
            all_err.append(err)
    return np.average(all_err)


def err_percent(error_rate):
    return str(100 * error_rate) + "%"


def compute():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # data_path = os.path.join(dir_path, "remy_data_all.csv")
    data_path = os.path.join(dir_path, "remy_data_cleaned_with_header.csv")
    data_all = pd.read_csv(data_path, header=0)
    labels = np.asarray(data_all.iloc[:, 0], dtype=np.int)
    nr_class = len(np.unique(labels))
    X = np.asarray(data_all.iloc[:, 1:], dtype=np.float)
    y = labels
    # print('X: ', X)
    # print('y: ', y)
    print('size of X: ', X.shape)
    print('size of y: ', y.shape)

    # remove the dependent columns
    # Q, R = qr(a=X, mode='reduced')

    print('desriptive statistics for X: ', describe(X))

    # print('X affine: ', X)

    X_short = X[:, :X.shape[0]]
    mean = np.mean(X_short, axis=0)
    std = np.std(X_short, axis=0)
    X_norm = (X_short - mean) / std

    # make the classifier affine
    ones = np.ones(X_norm.shape[0])
    X_norm = np.concatenate((ones[:, np.newaxis], X_norm), axis=1)

    w_hat = find_w(X_norm, y)
    y_hat = np.rint(np.matmul(X_norm, w_hat))
    print("check y_hat: ", y_hat)
    diff = np.sum(y_hat != y)
    err_total = diff / len(y)
    print('Error on the whole data: ',
          err_percent(err_total))

    err_total = cross_validate(X, y, nr_class=nr_class, is_affine=True)
    print('Error affine classifier on standard cross-validation: ',
          err_percent(err_total))

    X_lin = X[:, 1:]
    err_total = cross_validate(X_lin, y, nr_class)
    print('Error linear classifier on standard cross-validation: ',
          err_percent(err_total))

    nr_samples = [5] + [x * 10 for x in range(1, 13)]
    error_rates = []
    for nr_sample in nr_samples:
        err_total = cross_validate(X_lin, y, train_limit=nr_sample,
                                   nr_class=nr_class)
        error_rates.append(err_total)
    print('# of train samples: ', nr_samples)
    print('error rates: ', error_rates)
    plot(nr_samples=nr_samples, error_rates=error_rates)

    X_lin_3 = X[:, 1:4]
    err_total = cross_validate(X_lin_3, y)
    print('Error linear classifier on standard'
          'cross-validation on only the first 3 features: ',
          err_percent(err_total))


if __name__ == "__main__":
    compute()
