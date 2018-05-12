# As usual, a bit of setup
from __future__ import print_function

from cs231n.layers import *

import time

from scipy import signal


def reshape(x):
    return x.reshape(1, 1, 1, -1)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


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
    return conv_forward_naive(reshape(x), reshape(filters), b, conv_param)[0]


def conv_fft():
    return conv_forward_fft(reshape(x), reshape(filters), b, conv_param)[0]


def timeit(statement, number=1):
    t0 = time.time()
    for _ in range(number):
        result = statement()
    t1 = time.time()
    return t1 - t0, result


conv_param = {'stride': stride, 'pad': padding}
conv_naive_time, result_naive = timeit(conv_naive, number=exec_number)
print("result naive: ", result_naive)
print("conv naive time: ", conv_naive_time)
conv_fft_time, result_fft = timeit(conv_fft, number=exec_number)
print("result_fft: ", result_fft)
are_close = np.allclose(result_fft, result_naive)
print("conv fft time: ", conv_fft_time, ",are close: ", are_close)
numpy_time, _ = timeit(wrapper(np.correlate, x, filters, mode=mode), number=exec_number)
print("numpy time: ", numpy_time)
scipy_time, _ = timeit(wrapper(signal.correlate, x, filters, mode=mode), number=exec_number)
print("scipy time: ", scipy_time)

with open("conv_timimg" + str(time.time()) + ".csv", "w+") as out_file:
    out_file.write("filter_size, naive time (sec), fft time (sec), fast time (sec), fftw time (sec)\n")
    for filter_size in range(1, 65):
        filters = np.random.randn(filter_size)
        conv_naive_time, _ = timeit(wrapper(conv_forward_naive_1D, reshape(x), reshape(filters), b, conv_param),
                                    number=exec_number)
        conv_fft_time, _ = timeit(wrapper(conv_forward_fft_1D, reshape(x), reshape(filters), b, conv_param), number=exec_number)
        numpy_time, _ = timeit(wrapper(np.correlate, x, filters, mode=mode), number=exec_number)
        scipy_time, _ = timeit(wrapper(signal.correlate, x, filters, mode=mode), number=exec_number)
        result = [filter_size, conv_naive_time, conv_fft_time, numpy_time, scipy_time]
        out_file.write(",".join([str(x) for x in result]) + "\n")
