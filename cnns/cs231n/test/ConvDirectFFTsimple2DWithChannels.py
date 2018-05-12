# As usual, a bit of setup
from __future__ import print_function

from cs231n.classifiers.cnn import *
from cs231n.layers import *
from cs231n.fast_layers import *

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


print("simple direct and FFT convolution for 2D")

x1 = np.array([[3, 1, 2], [4, 5, 7], [1, 9, 0]])
x2 = np.array([[1, 2, 3], [5, 0, 3], [2, 8, 8]])
x = np.array([x1, x2])

filter1a = np.array([[1, 2], [0, 1]])
filter1b = np.array([[5, 0], [1, 3]])

filter2a = np.array([[2, 2], [3, 3]])
filter2b = np.array([[5, 5], [4, 4]])
filters = np.array([[filter1a, filter1b], [filter2a, filter2b]])

b = np.array([3, 0])

# reshape to have 4 dimensions (number of inputs in the mini-batch, # of channels, height, width of the input
x = x.reshape(1, 2, 3, 3)
filters = filters.reshape(2, 2, 2, 2)

padding = 0
stride = 1

conv_param = {'stride': stride, 'pad': padding}
outnaive, _ = conv_forward_naive(x, filters, b, conv_param)
print("out naive conv: ", outnaive)
print("out naive conv shape: ", outnaive.shape)

outfast, _ = conv_forward_fast(x, filters, b, conv_param)
print("outfast: ", outfast)

outfft, _ = conv_forward_fft(x, filters, b, conv_param)
print("out fft conv: ", outfft)

print("is the fft cross_correlation for convolution correct with respect to naive: ", np.allclose(outfft, outnaive, atol=1e-12))
print("absolute error fft naive: ", np.sum(np.abs(outfft - outnaive)))
print("relative error fft naive: ", rel_error(outfft, outnaive))

print("is the fft cross_correlation for convolution correct with respect to fast: ", np.allclose(outfft, outfast, atol=1e-12))
print("absolute error fft fast: ", np.sum(np.abs(outfft - outfast)))
print("relative error fft fast: ", rel_error(outfft, outfast))
