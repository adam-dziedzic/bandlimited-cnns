import numpy as np
from scipy import signal
from cs231n.layers import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


x = [1, 2, 3, 5, 1, -1, 2, 3, 5, 8, 3, 9, 1, 2, 5, 1]
x = np.array(x)
filters = [4, 5, 3, 4]
filters = np.array(filters)

b = np.array([0])

standard_conv = signal.convolve(x, filters, 'valid')
print("conv scipy:", standard_conv)

x = x.reshape(1, 1, -1)
filters = filters.reshape(1, 1, -1)

stride = 1
padding = 0
conv_param = {'stride': stride, 'pad': padding}
outnaive, _ = conv_forward_naive_1D(x, filters, b, conv_param)
print("out naive conv: ", outnaive)
print("out naive conv shape: ", outnaive.shape)

outfft, _ = conv_forward_fft_1D(x, filters, b, conv_param)
print("outfff: ", outfft)
print("outfff shape: ", outfft.shape)

print("is the fft cross_correlation for convolution correct with respect to naive: ",
      np.allclose(outfft, outnaive, atol=1e-12))
print("absolute error fft naive: ", np.sum(np.abs(outfft - outnaive)))
print("relative error fft naive: ", rel_error(outfft, outnaive))
