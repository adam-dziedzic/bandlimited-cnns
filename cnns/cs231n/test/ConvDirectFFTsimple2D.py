# As usual, a bit of setup
from __future__ import print_function

from cs231n.classifiers.cnn import *
from cs231n.layers import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


print("simple direct and FFT convolution for 2D")

x = np.array([[3, 1, 2],
              [4, 5, 7],
              [1, 9, 0]])
filter = np.array([[1, 2], [0, 1]])
b = np.array([0])

# reshape to have 4 dimensions (number of inputs in the mini-batch, # of channels, height, width of the input
x = x.reshape(1, 1, 3, 3)
filter = filter.reshape(1, 1, 2, 2)

padding = 0
stride = 1

conv_param = {'stride': stride, 'pad': padding}
outnaive, _ = conv_forward_naive(x, filter, b, conv_param)
print("out naive conv: ", outnaive)

outfft, _ = conv_forward_fft(x, filter, b, conv_param)
print("out fft conv: ", outfft)

print("is the fft cross_correlation for convolution correct: ", np.allclose(outfft, outnaive, atol=1e-12))
print("absolute error: ", np.sum(np.abs(outfft - outnaive)))
print("relative error: ", rel_error(outfft, outnaive))

outfftw, _ = conv_forward_fftw(x, filter, b, conv_param)
print("out fftw conv: ", outfftw)

print("is the fftw cross_correlation for convolution correct: ", np.allclose(outfftw, outnaive, atol=1e-12))
print("absolute error fftw: ", np.sum(np.abs(outfftw - outnaive)))
print("relative error fftw: ", rel_error(outfftw, outnaive))