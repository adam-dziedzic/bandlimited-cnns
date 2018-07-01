# As usual, a bit of setup
from __future__ import print_function

from nnlib.classifiers.cnn import *
from nnlib.layers import *

import timeit
import time

from scipy import signal


def reshape(x):
    return x.reshape(1, 1, 1, -1)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def _template_func(setup, func):
    """
    We want to access the result of the called function (not only its timing).
    Create a timer function. Used if the "statement" is a callable.
    """

    def inner(_it, _timer, _func=func):
        setup()
        _t0 = _timer()
        for _i in _it:
            retval = _func()
        _t1 = _timer()
        return _t1 - _t0, retval

    return inner


timeit._template_func = _template_func
timeit.template = """
def inner(_it, _timer, _func=func):
    setup()
    _t0 = _timer()
    for _i in _it:
        retval = _func()
    _t1 = _timer()
    return _t1 - _t0, retval
"""

print("timeit: simple direct and FFT convolution for 1D")

np.random.seed(231)

num_channels = 1

input_size = 64
filter_size = 4

x = np.random.randn(input_size)
filters = np.random.randn(filter_size)

b = np.random.randn(1)

stride = 1

mode = "valid"
if mode == "valid":
    padding = 0
elif mode == "full":
    padding = len(filters) - 1

exec_number = 1  # number which is the number of executions you'd like to run


# decorator - to time the functions with arguments
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def conv_naive():
    conv_forward_naive(reshape(x), reshape(filters), b, conv_param)


def conv_fft():
    conv_forward_fft(reshape(x), reshape(filters), b, conv_param)


def ttimeit(statement, number):
    print(timeit)
    return timeit.Timer(statement).timeit(number)


conv_param = {'stride': stride, 'pad': padding}
conv_naive_time = ttimeit(conv_naive, number=exec_number)
print("conv naive time: ", conv_naive_time)
conv_fft_time, _ = ttimeit(conv_fft, number=exec_number)
print("conv fft time: ", conv_fft_time)
numpy_time, _ = timeit.timeit(wrapper(np.correlate, x, filters, mode=mode), number=exec_number)
print("numpy time: ", numpy_time)
scipy_time, _ = timeit.timeit(wrapper(signal.correlate, x, filters, mode=mode), number=exec_number)
print("scipy time: ", scipy_time)

with open("conv_timimg" + str(time.time()) + ".csv", "w+") as out_file:
    out_file.write("filter_size, naive time (sec), fft time (sec), fast time (sec), fftw time (sec)\n")
    for filter_size in range(1, 65):
        filters = np.random.randn(filter_size)
        conv_naive_time, _ = timeit.timeit(wrapper(conv_forward_naive_1D, x, filters, b, conv_param),
                                           number=exec_number)
        conv_fft_time, _ = timeit.timeit(wrapper(conv_forward_fft_1D, x, filters, b, conv_param), number=exec_number)
        numpy_time, _ = timeit.timeit(wrapper(np.correlate, x, filters, mode=mode), number=exec_number)
        scipy_time, _ = timeit.timeit(wrapper(signal.correlate, x, filters, mode=mode), number=exec_number)
        result = [filter_size, conv_naive_time, conv_fft_time, numpy_time, scipy_time]
        out_file.write(",".join([str(x) for x in result]) + "\n")
