import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pickle
from enum import Enum


def get_log_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


additional_log_file = "additional-info-" + get_log_time() + ".log"
mem_log_file = "mem_log-" + get_log_time() + ".log"


class EnumWithNames(Enum):
    """
    The Enum classes that inherit from the EnumWithNames will get the get_names
    method to return an array of strings representing all possible enum values.
    """

    @classmethod
    def get_names(cls):
        return [enum_value.name for enum_value in cls]


class OptimizerType(EnumWithNames):
    MOMENTUM = 1
    ADAM = 2
    SGD = 3


class MemoryType(EnumWithNames):
    STANDARD = 1
    PINNED = 2


class RunType(EnumWithNames):
    TEST = 0
    DEBUG = 1


class ModelType(EnumWithNames):
    RES_NET = 0
    DENSE_NET = 1


DEFAULT_OPTIMIZER = OptimizerType.ADAM
DEFAULT_RUN_TYPE = RunType.TEST
DEFAULT_MODEL_TYPE = ModelType.DENSE_NET


class ConvType(EnumWithNames):
    STANDARD = 1
    SPECTRAL_PARAM = 2
    SPECTRAL_DIRECT = 3
    SPATIAL_PARAM = 4
    FFT1D = 5
    AUTOGRAD = 6
    SIMPLE_FFT = 7
    SIMPLE_FFT_FOR_LOOP = 8
    COMPRESS_INPUT_ONLY = 9


class CompressType(EnumWithNames):
    """

    >>> compress_type = CompressType.BIG_COEFF
    >>> assert compress_type.value == 2
    >>> compress_type = CompressType(3)
    >>> assert compress_type is CompressType.LOW_COEFF
    """
    # Compress both the input signal and the filter.
    STANDARD = 1
    # Preserve the largest (biggest) coefficients and zero-out the lowest
    # (smallest) coefficients.
    BIG_COEFF = 2
    # Preserve only the lowest coefficients but zero-out the highest
    # coefficients.
    LOW_COEFF = 3
    # Compress the filters for fft based convolution or only the input signals.
    NO_FILTER = 4


class NetworkType(EnumWithNames):
    STANDARD = 1
    SMALL = 2


def energy(x):
    """
    Calculate the energy of the signal.

    :param x: the input signal
    :return: energy of x
    """
    return np.sum(np.power(x, 2))


def next_power2(x):
    """
    :param x: an integer number
    :return: the power of 2 which is the larger than x but the smallest possible
    """
    return 2 ** np.ceil(np.log2(x)).astype(int)


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


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output)


def plot_signal(signal, title="signal", xlabel="Time"):
    plt.plot(range(0, len(signal)), signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    plt.show()


def plot_signal_freq(signal, title="signal", xlabel="Frequency"):
    plot_signal(signal, title, xlabel)


def plot_signals(x, y, title="", xlabel="Time", ylabel="Amplitude",
                 label_x="input", label_y="output",
                 linestyle="solid"):
    fontsize = 20
    linewidth = 2.0
    plt.plot(range(0, len(x)), x, color="red", linewidth=linewidth,
             linestyle=linestyle)
    plt.plot(range(0, len(y)), y, color="blue", linewidth=linewidth,
             linestyle=linestyle)
    # We prepare the plot
    fig = plt.figure(1)
    plot = fig.add_subplot(111)
    # We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=fontsize)
    plot.tick_params(axis='both', which='minor', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    red_patch = mpatches.Patch(color='red', label=label_x)
    blue_patch = mpatches.Patch(color='blue', label=label_y)
    plt.legend(handles=[red_patch, blue_patch], fontsize=fontsize,
               loc='upper left')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
