import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pickle


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


def get_log_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


def plot_signal(signal, title="signal"):
    plt.plot(range(0, len(signal)), signal)
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_signals(x, y, title="", xlabel="Time", ylabel="Amplitude", label_x="input", label_y="output",
                 linestyle="solid"):
    fontsize = 20
    linewidth = 2.0
    plt.plot(range(0, len(x)), x, color="red", linewidth=linewidth, linestyle=linestyle)
    plt.plot(range(0, len(y)), y, color="blue", linewidth=linewidth, linestyle=linestyle)
    # We prepare the plot
    fig = plt.figure(1)
    plot = fig.add_subplot(111)
    # We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=fontsize)
    plot.tick_params(axis='both', which='minor', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    red_patch = mpatches.Patch(color='red', label=label_x)
    blue_patch = mpatches.Patch(color='blue', label=label_y)
    plt.legend(handles=[red_patch, blue_patch], fontsize=fontsize, loc='upper left')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
