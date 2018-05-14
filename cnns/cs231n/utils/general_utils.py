import numpy as np


def reshape_3d_rest(x):
    """
    Reshape the one dimensional input x into a 3 dimensions, with the 1 for each of the first 2 dimensions and the last
    dimension with all the elements of x.

    :param x: a 1D input array
    :return: reshaped array to 3 dimensions with the last one having all the input values
    """
    return x.reshape(1, 1, -1)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def abs_error(x, y):
    """ returns the absolute error """
    return np.sum(np.abs(x - y))
