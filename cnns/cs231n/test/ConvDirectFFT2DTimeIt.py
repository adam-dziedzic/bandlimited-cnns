# As usual, a bit of setup
from __future__ import print_function

from cs231n.classifiers.cnn import *
from cs231n.layers import *

import timeit
import time

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


print("timeit: simple direct and FFT convolution for 2D")

np.random.seed(231)

num_channels = 3

input_dim = (num_channels, 64, 64)
filter_size = 4
filter_dim = (num_channels, filter_size, filter_size)
num_inputs = 10
num_filters = 5

x = np.random.randn(num_inputs, *input_dim)
filters = np.random.randn(num_filters, *filter_dim)

b = np.random.randn(num_filters)

padding = 0
stride = 1

exec_number = 1  # number which is the number of executions you'd like to run the

# decorator - to time the functions with arguments
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def conv_naive():
    conv_forward_naive(x, filters, b, conv_param)


def conv_fft():
    conv_forward_fft(x, filters, b, conv_param)


conv_param = {'stride': stride, 'pad': padding}
conv_naive_time = timeit.timeit(conv_naive, number=exec_number)
print("conv naive time: ", conv_naive_time)
conv_fft_time = timeit.timeit(conv_fft, number=exec_number)
print("conv fft time: ", conv_fft_time)
conv_fast_time = timeit.timeit(wrapper(conv_forward_fast, x, filters, b, conv_param), number=exec_number)
print("conv fast time: ", conv_fast_time)
conv_fftw_time = timeit.timeit(wrapper(conv_forward_fftw, x, filters, b, conv_param), number=exec_number)
print("conv fftw time: ", conv_fftw_time)

with open("conv_timimg"+str(time.time())+".csv", "w+") as out_file:
    out_file.write("filter_size, naive time (sec), fft time (sec), fast time (sec), fftw time (sec)\n")
    for filter_size in range(1, 65):
        filters = np.random.randn(num_filters, num_channels, filter_size, filter_size)
        conv_naive_time = timeit.timeit(wrapper(conv_forward_naive, x, filters, b, conv_param), number=exec_number)
        conv_fft_time = timeit.timeit(wrapper(conv_forward_fft, x, filters, b, conv_param), number=exec_number)
        conv_fast_time = timeit.timeit(wrapper(conv_forward_fast, x, filters, b, conv_param), number=exec_number)
        conv_fftw_time = timeit.timeit(wrapper(conv_forward_fftw, x, filters, b, conv_param), number=exec_number)
        result = [filter_size, conv_naive_time, conv_fft_time, conv_fast_time, conv_fftw_time]
        out_file.write(",".join([str(x) for x in result]) + "\n")
