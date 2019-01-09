"""
Custom FFT based convolution that can rely on the autograd
(a tape-based automatic differentiation library that supports
all differentiable Tensor operations in pytorch).
"""

import logging
import torch
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftAutograd
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor

import conv1D_cuda

torch.manual_seed(31)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class Conv1dfftFunctionCuda(torch.autograd.Function):
    """
    Implement the 1D convolution via FFT with compression of the input map and
    the filter.
    """

    @staticmethod
    # @profile
    def forward(ctx, input, filter, bias=None, padding=None, index_back=None):
        """
        Compute the forward pass for the 1D convolution.

        :param ctx: context to save intermediate results, in other
        words, a context object that can be used to stash information
        for backward computation
        :param input: the input map to the convolution (e.g. a time-series).

        The other parameters are similar to the ones in the
        Conv2dfftAutograd class.

        :param filter: the filter (a.k.a. kernel of the convolution).
        :param bias: the bias term for each filter.
        :param padding: how much pad each end of the input signal.
        :param index_back: how many last elements in the fft-ed signal to
        discard.
        """
        outputs = conv1D_cuda(input, filter, bias, padding, index_back)
        output = outputs[0]
        xfft, yfft, W, WW, fft_size = outputs[1:]
        if ctx:
            ctx.W = W
            ctx.WW = WW
            ctx.fft_size = fft_size
            ctx.save_for_backward(xfft, yfft)
        return output

    @staticmethod
    def backward(ctx, dout):
        """
        Compute the gradient using FFT.

        Requirements from PyTorch: backward() - gradient formula.
        It will be given as many Variable arguments as there were
        outputs, with each of them representing gradient w.r.t. that
        output. It should return as many Variables as there were
        inputs, with each of them containing the gradient w.r.t. its
        corresponding input. If your inputs did not require gradient
        (see needs_input_grad), or were non-Variable objects, you can
        return None. Also, if you have optional arguments to forward()
        you can return more gradients than there were inputs, as long
        as they’re all None.
        In short, backward() should return as many tensors, as there
        were inputs to forward().

        :param ctx: context with saved variables
        :param dout: output gradient
        :return: gradients for input map x, filter w and bias b
        """
        xfft, yfft = ctx.saved_tensors
        W = ctx.W
        WW = ctx.WW
        fft_size = ctx.fft_size

        # The last dimension (_) for xfft and yfft is the 2 element complex
        # number.
        N, C, half_fft_compressed_size, _ = xfft.shape
        F, C, half_fft_compressed_size, _ = yfft.shape
        N, F, out_W = dout.shape

        print("W: ", W, ", WW: ", WW, ", fft_size: ", fft_size, ",N: ", N,
              ", F:", F, ", out_W: ", out_W)


class Conv1dfftCuda(Conv1dfftAutograd):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=None, padding=None, bias=True, index_back=None,
                 out_size=None, filter_value=None, bias_value=None,
                 use_next_power2=False, conv_index=None, preserve_energy=None,
                 is_debug=True, compress_filter=True, big_coef=False):
        super(Conv1dfftCuda, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            index_back=index_back, out_size=out_size, filter_value=filter_value,
            bias_value=bias_value, use_next_power2=use_next_power2,
            conv_index=conv_index, preserve_energy=preserve_energy,
            is_debug=is_debug, compress_filter=compress_filter,
            big_coef=big_coef)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return Conv1dfftFunctionCuda.apply(
            input, self.filter, self.bias, self.padding, self.compress_rate)
