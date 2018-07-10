"""
Custom FFT based convolution that relies on the autograd (a tape-based automatic differentiation library that supports
all differentiable Tensor operations in pytorch).
"""

import math

import logging
import numpy as np
import torch
from torch import cat, mul, add, tensor
from torch.nn import Module
from torch.nn.functional import pad
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


def next_power2(x):
    """
    :param x: an integer number
    :return: the power of 2 which is the larger than x but the smallest possible

    >>> result = next_power2(5)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(1)
    >>> np.testing.assert_equal(result, 1)
    >>> result = next_power2(2)
    >>> np.testing.assert_equal(result, 2)
    >>> result = next_power2(7)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(9)
    >>> np.testing.assert_equal(result, 16)
    """
    return math.pow(2, math.ceil(math.log2(x)))


def complex_mul(x, y):
    """
    Multiply arrays of complex numbers (it also handles multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural Networks (https://arxiv.org/pdf/1509.09308.pdf)
    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.], [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.], [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul(x, y), tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.], [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul(x, y), tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))
    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))
    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    """
    ua = x.narrow(-1, 0, 1)
    ud = x.narrow(-1, 1, 1)
    va = y.narrow(-1, 0, 1)
    vb = y.narrow(-1, 1, 1)
    ub = add(ua, ud)
    uc = add(ud, mul(ua, -1))
    vc = add(va, vb)
    uavc = mul(ua, vc)
    result_rel = add(uavc, mul(mul(ub, vb), -1))  # relational part of the complex number
    result_im = add(mul(uc, va), uavc)  # imaginary part of the complex number
    result = cat(seq=(result_rel, result_im), dim=-1)  # use the last dimension
    return result


def pytorch_conjugate(x):
    """
    Conjugate all the complex numbers in tensor x in place.

    :param x: PyTorch tensor with complex numbers
    :return: conjugated numbers in x

    >>> x = tensor([[1, 2]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2]]))
    >>> x = tensor([[1, 2], [3, 4]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2], [3, -4]]))
    >>> x = tensor([[[1, 2], [3, 4]], [[0.0, 0.0], [0., 1.]]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[[1, -2], [3, -4]], [[0., 0.], [0., -1]]]))
    """
    x.narrow(-1, 1, 1).mul_(-1)
    return x


def get_full_energy(x):
    """
    Return the full energy of the signal. The energy E(xfft) of a sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x

    >>> x = torch.tensor([1.2, 1.0])
    >>> full_energy, squared = get_full_energy(x)
    >>> np.testing.assert_almost_equal(full_energy, 2.4400, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, torch.tensor([2.4400]))

    >>> x_torch = torch.tensor([[1.2, 1.0], [0.5, 1.4]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    >>> x_torch = torch.tensor([[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)
    """
    # the signal in frequency domain is symmetric and pytorch already discards second half of the signal
    squared = torch.add(torch.pow(x.narrow(-1, 0, 1), 2), torch.pow(x.narrow(-1, 1, 1), 2)).squeeze()
    full_energy = torch.sum(squared).item()  # sum of squared values of the signal
    return full_energy, squared


def preserve_energy_index(xfft, energy_rate=None, index_back=None):
    """
    To which index should we preserve the xfft signal (and discard the remaining coefficients). This is based on the
    provided energy_rate, or if the energy_rate is not provided, then index_back is applied. If none of the params are
    provided, we

    :param xfft: the input signal to be truncated
    :param energy_rate: how much energy should we preserve in the xfft signal
    :param index_back: how many coefficients of xfft should we discard counting from the back of the xfft
    :return: calculated index, no truncation applied to xfft itself, returned at least the index of first coefficient
    >>> xfft = torch.tensor([])
    >>> result = preserve_energy_index(xfft, energy_rate=1.0)
    >>> np.testing.assert_equal(result, None)

    >>> xfft = torch.tensor([[1., 2.], [3., 4.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=1.0)
    >>> np.testing.assert_equal(result, 3)

    >>> xfft = [[1., 2.], [3., 4.]]
    >>> result = preserve_energy_index(xfft)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = [[1., 2.], [3., 4.], [5., 6.]]
    >>> result = preserve_energy_index(xfft, index_back=1)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = torch.tensor([[1., 2.], [3., 4.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.9)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.5)
    >>> np.testing.assert_equal(result, 1)

    >>> xfft = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.0)
    >>> np.testing.assert_equal(result, 1)

    >>> x_torch = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> squared_numpy = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> full_energy_numpy = np.sum(squared_numpy)
    >>> # set energy rate to preserve only the first and second coefficients
    >>> energy_rate = squared_numpy[0]/full_energy_numpy + 0.0001
    >>> result = preserve_energy_index(x_torch, energy_rate=energy_rate)
    >>> np.testing.assert_equal(result, 2)
    >>> # set energy rate to preserved only the first coefficient
    >>> energy_rate = squared_numpy[0]/full_energy_numpy - 0.0001
    >>> result = preserve_energy_index(x_torch, energy_rate=energy_rate)
    >>> np.testing.assert_equal(result, 1)
    """
    if xfft is None or len(xfft) == 0:
        return None
    if energy_rate is not None:
        full_energy, squared = get_full_energy(xfft)
        current_energy = 0.0
        preserved_energy = full_energy * energy_rate
        index = 0
        while current_energy < preserved_energy and index < len(squared):
            current_energy += squared[index]
            index += 1
        return max(index, 1)
    elif index_back is not None:
        return len(xfft) - index_back
    return len(xfft)


def correlate_signals(x, y, fft_size, out_size, preserve_energy_rate=None, index_back=None, signal_ndim=1):
    """
    Cross-correlation of the signals: x and y.

    :param x: input signal
    :param y: filter
    :param out_len: required output len
    :param preserve_energy_rate: compressed to this energy rate
    :param index_back: how many coefficients to remove
    :return: output signal after correlation of signals x and y
    """
    xfft = torch.rfft(x, signal_ndim)
    yfft = torch.rfft(y, signal_ndim)
    if preserve_energy_rate is not None or index_back is not None:
        index = preserve_energy_index(xfft, preserve_energy_rate, index_back)
        # with open(log_file, "a+") as f:
        #     f.write("index: " + str(index_back) + ";preserved energy input: " + str(
        #         compute_energy(xfft[:index]) / compute_energy(xfft[:fft_size // 2 + 1])) +
        #             ";preserved energy filter: " + str(
        #         compute_energy(yfft[:index]) / compute_energy(yfft[:fft_size // 2 + 1])) + "\n")
        xfft = xfft.narrow(-1, 0, index)
        yfft = yfft.narrow(-1, 0, index)
        # print("the signal size after compression: ", index)
        xfft = xfft.pad(input=xfft, pad=(0, fft_size - index), mode='constant', value=0)
        yfft = yfft.pad(input=yfft, pad=(0, fft_size - index), mode='constant', value=0)
    out = torch.irfft(complex_mul(xfft, pytorch_conjugate(yfft)))

    # plot_signal(out, "out after ifft")
    out = out[:out_size]
    # plot_signal(out, "after truncating to xlen: " + str(x_len))
    return out


class PyTorchConv1d(Module):
    def __init__(self, filter_width, filter=None, bias=None, padding=None, preserve_energy_rate=None):
        """
        1D convolution using FFT implemented fully in PyTorch.

        :param filter_width: the width of the filter
        :param filter_height: the height of the filter
        :param filter: you can provide the initial filter, i.e. filter weights of shape (F, C, WW), where
        F - number of filters, C - number of channels, WW - size of the filter
        :param bias: you can provide the initial value of the bias, of shape (F,)
        :param padding: the padding added to the front and back of the input signal
        :param preserve_energy_rate: the energy of the input to the convolution (both, the input map and filter) that
        have to be preserved (the final length is the length of the longer signal that preserves the set energy rate).

        Regarding, the stride parameter: the number of pixels between adjacent receptive fields in the horizontal and
        vertical directions, it is 1 for the FFT based convolution.
        """
        super(PyTorchConv1d, self).__init__()
        if filter is None:
            self.filter = Parameter(torch.randn(1, 1, filter_width))
        else:
            self.filter = filter
        if bias is None:
            self.bias = Parameter(torch.randn(1))
        else:
            self.bias = bias
        self.padding = padding
        self.preserve_energy_rate = preserve_energy_rate

    def forward(self, input):
        """
        Forward pass of 1D convolution.

        The input consists of N data points with each data point representing a signal (e.g., time-series) of length W.

        We also have the notion of channels in the 1-D convolution. We want to use more than a single filter even for
        the input signal, so the output is a batch with the same size but the number of output channels is equal to the
        number of input filters.

        :param x: Input data of shape (N, C, W), N - number of data points in the batch, C - number of channels, W - the
        width of the signal or time-series (number of data points in a univariate series)
        :param w:
        :param b: biases
        :param conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.
        :return: output data, of shape (N, F, W') where W' is given by: W' = 1 + (W + 2*pad - WW)

         :see:  source short: https://goo.gl/GwyhXz
         full: https://stackoverflow.com/questions/40703751/
         using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        """
        N, C, W = input.size()
        F, C, WW = self.filter.size()
        fftsize = next_power2(W + WW - 1)
        # pad only the dimensions for the time-series (and neither data points nor the channels)
        padded_x = pad(input, (self.padding, self.padding), 'constant', 0)

        out_W = W + 2 * pad - WW + 1
        out = torch.empty([N, F, out_W])
        for nn in range(N):  # For each time-series in the input batch.
            for ff in range(F):  # For each filter in w
                for cc in range(C):
                    out[nn, ff] += correlate_signals(padded_x[nn, cc], self.filter[ff, cc], fftsize,
                                                     out_size=out_W, preserve_energy_rate=self.preserve_energy_rate,
                                                     index_back=self.index_back)
                out[nn, ff] += self.bias[ff]  # add the bias term

        return out


def test_run():
    torch.manual_seed(231)
    module = PyTorchConv1d(3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)


if __name__ == "__main__":
    import sys
    import doctest

    test_run()
    sys.exit(doctest.testmod()[0])

"""
expected gradient for the input:  tensor([[[[-0.2478, -0.7275, -1.0670,  1.3629,  1.7458, -0.5786, -0.2722,
          0.5767,  0.7379, -0.6335],
        [-1.7113,  0.1839,  1.0434, -3.5176, -1.7056,  1.0892,  2.0054,
          2.3190, -1.6143, -1.3427],
        [-2.4303, -0.1218,  1.9863, -1.6753, -0.3529, -2.4454,  0.4331,
          1.8996,  1.5348, -0.3813],
        [-1.7727, -1.4130,  2.8780, -0.1220, -1.1942,  0.9997, -2.8926,
         -1.4083,  1.1635,  0.9641],
        [ 0.2487,  0.0023,  0.3793, -0.4038,  1.3017,  0.1421, -0.9947,
          0.5084,  0.1511, -2.1860],
        [-0.1263,  1.7602,  3.3994,  0.7883,  0.6831, -0.7291, -0.3211,
          1.8856,  0.3729, -1.2780],
        [-2.1050,  1.8296,  2.4018,  0.5756,  1.3364, -2.9692, -0.4314,
          3.3727,  3.1612, -1.0387],
        [-0.5624, -1.0603,  0.8454,  0.2767,  0.3005,  0.3977, -1.1085,
         -2.7611, -0.4906, -0.1018],
        [ 0.4603, -0.7684,  1.0566, -0.8825,  0.8468,  1.0482,  1.2088,
          0.2836,  0.0993, -0.0322],
        [ 0.0131,  0.4351, -0.3529,  0.2088, -0.3471,  0.3255,  1.6812,
          0.1925, -0.6875,  0.1037]]]])
"""
