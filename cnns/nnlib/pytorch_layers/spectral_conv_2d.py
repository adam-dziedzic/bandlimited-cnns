import math

import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


"""
Spectral convolutional layer with parameter (convolution kernels) 
initialization in the spectral domain. 
"""


def _glorot_sample(kernel_size, in_channels, out_channels, transposed=False):
    """
    The definition of the glorot initialization but not for a variable but for
    a separate sample.

    :param kernel_size: The width and length of the filter (kernel).
    :param in_channels: Number of input channels for an image (typically 3
      for RGB or 1 for a gray scale image).
    :param out_channels: Number of filters in a layer
    :return: numpy array with glorot initialized values
    """
    limit = np.sqrt(6 / (in_channels + out_channels))
    if transposed:
        size = (in_channels, out_channels, kernel_size, kernel_size)
    else:
        size = (out_channels, in_channels, kernel_size, kernel_size)
    return np.random.uniform(low=-limit, high=limit, size=size)


class SpectralConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, transposed=False):
        """

        :param in_channels: number of input channels for the convolution.
        :param out_channels: number of filter and the depth of the output
        :param kernel_size: the size of the kernel's width or height, the value
        is extended to a pair of numbers (kernel_size, kernel_size).
        :param stride: the stride size of the convolution
        :param padding: padding of the input
        :param dilation: dilation of the convolution (see the original
        convolution in pytorch for more details)
        :param groups: (see the original
        convolution in pytorch for more details)
        :param bias: should add bias to each filter
        :param transposed: (see the original
        convolution in pytorch for more details)
        """
        super(SpectralConv2d, self).__init__()
        print("use spectral convolution")
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.signal_ndim = 2
        self.kernel_size = kernel_size = _pair(kernel_size)
        if transposed:
            self.weight = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size)
            self.real = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.imag = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size)
            self.real = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.imag = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        """
        Run the convolution for the input map (e.g., the input image).

        :param input: input map
        :return: a feature map
        """
        spatial = torch.cat((self.real, self.imag), dim=-1)
        weight = torch.irfft(spatial, signal_ndim=self.signal_ndim,
                             onesided=True)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        """
        Reinitialized the parameters.
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        fft = torch.rfft(self.weight, signal_ndim=self.signal_ndim,
                         onesided=True)
        real = fft.narrow(-1, 0, 1)
        imag = fft.narrow(-1, 1, 1)
        self.real.data = torch.tensor(real)
        self.imag.data = torch.tensor(imag)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
