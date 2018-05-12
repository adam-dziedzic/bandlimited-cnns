import numpy as np
from scipy import signal
from cs231n.layers import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


x = [1, 2, 3, 5, 1, -1, 2, 3, 5, 8, 3, 9, 1, 2, 5, 1]
x = np.array(x)
print("input x shape: ", x.shape)
filters = [4, 5, 3, 4]
filters = np.array(filters)
b = np.array([0])

stride = 1
padding = len(filters)-1

scipy_correlate = signal.correlate(x, filters, 'full')
print("correlate scipy:", scipy_correlate)
print("correlate scipy shape:", scipy_correlate.shape)

np_correlate = np.correlate(x, filters, 'full')
print("correlate numpy:", np_correlate)
print("correlate numpy shape:", np_correlate.shape)

x = x.reshape(1, 1, -1)
filters = filters.reshape(1, 1, -1)

print("size of padding: ", padding)
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
