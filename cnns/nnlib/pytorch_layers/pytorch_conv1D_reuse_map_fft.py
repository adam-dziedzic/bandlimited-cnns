"""
Custom FFT based convolution that can rely on the autograd
(a tape-based automatic differentiation library that supports
all differentiable Tensor operations in pytorch).
"""
import logging
import sys

import numpy as np
import torch
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter

from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_fft_signals
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


def to_tensor(value):
    """
    `to_tensor` and `from_tensor` methods are used to transfer intermediate
    results between forward and backward pass of the convolution.

    Transform from None to -1 or retain the initial value
    for transition from a value or None to a tensor.

    :param value: a value to be changed to a tensor
    :return: a tensor representing the value, tensor with value -1 represents
    the None input
    """
    if value:
        return tensor([value])
    return tensor([-1])


def from_tensor(tensor_item):
    """
    `to_tensor` and `from_tensor` methods are used to transfer intermediate
    results between forward and backward pass of the convolution.

    Transform from tensor to a single numerical value or None (a tensor with
    value -1 is transformed to None).

    :param tensor_item: tensor with a single value
    :return: a single numerical value extracted from the tensor or None if the
    value is -1
    """
    value = tensor_item.item()
    if value == -1:
        return None
    return value


class PyTorchConv1dFunction(torch.autograd.Function):
    """
    Implement the 1D convolution via FFT with compression of the
    input map and the filter.
    """

    @staticmethod
    def forward(ctx, input, filter, bias, padding=None, index_back=None,
                out_size=None, signal_ndim=1):
        """
        Compute the forward pass for the 1D convolution.

        :param ctx: context to save intermediate results, in other
        words, a context object that can be used to stash information
        for backward computation
        :param input: the input map to the convolution (e.g. a time-series).

        The other parameters are similar to the ones in the
        PyTorchConv1dAutograd class.

        :param filter: the filter (a.k.a. kernel of the convolution)
        :param bias: the bias term for each filter
        :param padding: how much pad each end of the input signal
        :param index_back: how many last elements in the fft-ed signal to
        discard
        :param out_size: what is the expected output size - one can disard the
        elements in the frequency domain and do the spectral pooling within the
        convolution
        :param signal_ndim: this convolution is for 1 dimensional signals

        :return: the result of convolution
        """
        # N - number of input maps (or images in the batch),
        # C - number of input channels,
        # W - width of the input (the length of the time-series).
        N, C, W = input.size()

        # F - number of filters,
        # C - number of channels in each filter,
        # WW - the width of the filter (its length).
        F, C, WW = filter.size()

        out_W = W - WW + 1  # the length of the output (without padding)

        if padding is None:
            padding_count = 0
        else:
            padding_count = padding

        out_W += 2 * padding_count

        # We have to pad input with WW - 1 to execute fft correctly (no
        # overlapping signals) and optimize it by extending the signal to the
        # next power of 2. We want to reuse the fft-ed input x, so we use the
        # larger size chosen from: the filter width WW or output width out_W.
        # Larger padding does not hurt correctness of fft but make it slightly
        # slower, in terms of the computation time.
        WWW = max(out_W, WW)
        init_fft_size = next_power2(W + 2 * padding_count + WWW - 1)

        # How many padded (zero) values there are because of going to the next
        # power of 2?
        fft_padding_x = init_fft_size - 2 * padding_count - W

        # Pad only the dimensions for the time-series - the width dimension
        # (and neither data points nor the channels).

        padded_x = torch_pad(input,
                             (padding_count, padding_count + fft_padding_x),
                             'constant', 0)

        fft_padding_filter = init_fft_size - WW
        padded_filter = torch_pad(filter, (0, fft_padding_filter), 'constant',
                                  0)

        out = torch.zeros([N, F, out_W], dtype=input.dtype, device=input.device)

        # fft of the input and filters
        xfft = torch.rfft(padded_x, signal_ndim=signal_ndim, onesided=True)
        yfft = torch.rfft(padded_filter, signal_ndim=signal_ndim,
                          onesided=True)

        init_half_fft_size = xfft.shape[-1]

        # how much to compress the fft-ed signal
        half_fft_compressed_size = init_half_fft_size
        compress_fft_size = init_fft_size
        if out_size is not None:
            out_W = out_size
            compress_fft_size = out_size
            # We take onesided fft so the output after inverse fft should be out
            # size, thus the representation in spectral domain is twice smaller
            # than the one in time domain.
            half_fft_compressed_size = out_size // 2 + 1
            if index_back is not None:
                logger.error(
                    "index_back cannot be set when out_size is used")
                sys.exit(1)
        if index_back:
            half_fft_size = init_fft_size // 2 + 1
            half_fft_compressed_size = half_fft_size - index_back
            compress_fft_size = half_fft_compressed_size * 2

        # Complex numbers are represented as the pair of numbers in the last
        # dimension so we have to narrow the length of the last but one
        # dimension.
        if half_fft_compressed_size < init_half_fft_size:
            xfft = xfft.narrow(dim=-2, start=0, length=half_fft_compressed_size)
            yfft = yfft.narrow(dim=-2, start=0, length=half_fft_compressed_size)

        for nn in range(N):  # For each time-series in the batch.
            # Take one time series and unsqueeze it for broadcasting with
            # many filters.
            xfft_nn = xfft[nn].unsqueeze(0)
            out[nn] = correlate_fft_signals(
                xfft=xfft_nn, yfft=yfft, fft_size=compress_fft_size,
                out_size=out_W)
            # Add the bias term for each filter (it has to be unsqueezed to
            # the dimension of the out to properly sum up the values).
            out[nn] += bias.unsqueeze(1)

        if ctx:
            ctx.save_for_backward(
                xfft, yfft, bias, to_tensor(padding_count),
                to_tensor(W),
                to_tensor(WW),
                to_tensor(init_fft_size),
                to_tensor(compress_fft_size))

        return out

    @staticmethod
    def backward(ctx, dout):
        """
        Compute the gradient using FFT.

        Requirements from PyTorch: backward() - gradient formula.
        It will be given as many Variable arguments as there were
        outputs, with each of them representing gradient w.r.t. that
        output. It should return as many Variable s as there were
        inputs, with each of them containing the gradient w.r.t. its
        corresponding input. If your inputs didn’t require gradient
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
        logger.debug("execute backward")
        xfft, yfft, b, padding_count, W, WW, init_fft_size, compress_fft_size = ctx.saved_tensors
        padding_count = from_tensor(padding_count)
        W = from_tensor(W)
        WW = from_tensor(WW)
        init_fft_size = from_tensor(init_fft_size)
        compress_fft_size = from_tensor(compress_fft_size)
        signal_ndim = 1

        # The last dimension for xfft and yfft is the 2 element complex number.
        N, C, half_fft_compressed_size, _ = xfft.shape
        F, C, half_fft_compressed_size, _ = yfft.shape
        N, F, W_out = dout.shape

        dx = dw = db = None

        # Take the fft of dout (the gradient of the output of the forward pass).
        fft_padding_dout = init_fft_size - W_out
        padded_dout = torch_pad(dout, (0, fft_padding_dout), 'constant', 0)
        doutfft = torch.rfft(padded_dout, signal_ndim=signal_ndim,
                             onesided=True)
        initial_half_fft_size = doutfft.shape[-1]
        if half_fft_compressed_size < initial_half_fft_size:
            doutfft = doutfft.narrow(dim=-2, start=0,
                                     length=half_fft_compressed_size)

        if ctx.needs_input_grad[0]:
            # Initialize gradient output tensors.
            # the x used for convolution was with padding
            dx = torch.zeros([N, C, W], dtype=xfft.dtype)
            conjugate_yfft = pytorch_conjugate(yfft)
            for nn in range(N):
                # Take one time series and unsqueeze it for broadcast with
                # many gradients dout.
                doutfft_nn = doutfft[nn].unsqueeze(0)
                dx[nn] = correlate_fft_signals(
                    xfft=doutfft_nn, yfft=conjugate_yfft,
                    fft_size=compress_fft_size, out_size=W, is_forward=False)

        if ctx.needs_input_grad[1]:
            dw = torch.zeros([F, C, WW], dtype=yfft.dtype)
            # Calculate dw - the gradient for the filters w.
            # By chain rule dw is computed as: dout*x
            """
            More specifically:
            if the forward convolution is: [x1, x2, x3, x4] * [w1, w2], where *
            denotes the convolution operation, 
            Conv (out) = [x1 w1 + x2 w2, x2 w1 + x3 w2, x3 w1 + x4 w2]
            then the bacward to compute the 
            gradient for the weights is as follows (L - is the Loss function):
            gradient L / gradient w = 
            gradient L / gradient Conv x (times) gradient Conv / gradient w =
            dout x gradient Conv / gradient w = (^)
            
            gradient Conv / gradient w1 = [x1, x2, x3]
            gradient Conv / gradient w2 = [x2, x3, x4]
            
            dout = [dx1, dx2, dx3]
            
            gradient L / gradient w1 = dout * gradient Conv / gradient w1 =
            [dx1 x1 + dx2 x2 + dx3 x3]
            
            gradient L / gradient w2 = dout * gradient Conv / gradient w2 =
            [dx1 x2 + dx2 x3 + dx3 x4]
            
            Thus, the gradient for the weights is the convolution between the 
            flowing back gradient dout and the input x:
            gradient L / gradient w = [x1, x2, x3, x4] * [dx1, dx2, dx3]
            """
            for ff in range(F):
                doutfft_ff = doutfft[ff].unsqueeze(0)
                print("xfft: ", xfft)
                print("dout_fft: ", doutfft_ff)
                dw[ff] = correlate_fft_signals(
                    xfft=xfft, yfft=doutfft_ff, fft_size=compress_fft_size,
                    out_size=WW, signal_ndim=signal_ndim, is_forward=False)

        if ctx.needs_input_grad[2]:
            db = torch.zeros_like(b)

            # Calculate dB (the gradient for the bias term).
            # We sum up all the incoming gradients for each filter
            # bias (as in the affine layer).
            for ff in range(F):
                db[ff] += torch.sum(dout[:, ff, :])

        return dx, dw, db, None, None, None, None, None


class PyTorchConv1dAutograd(Module):
    def __init__(self, filter=None, bias=None, padding=None, index_back=None,
                 out_size=None, filter_width=None):
        """
        1D convolution using FFT implemented fully in PyTorch.

        :param filter: you can provide the initial filter, i.e.
        filter weights of shape (F, C, WW), where
        F - number of filters, C - number of channels, WW - size of
        the filter
        :param bias: you can provide the initial value of the bias,
        of shape (F,)
        :param padding: the padding added to the front and back of
        the input signal
        :param index_back: how many frequency coefficients should be
        discarded
        :param out_size: what is the expected output size of the
        operation (when compression is used and the out_size is
        smaller than the size of the input to the convolution, then
        the max pooling can be omitted and the compression
        in this layer can serve as the frequency-based (spectral)
        pooling.
        :param filter_width: the width of the filter

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, it has to be 1 for the FFT based convolution (at least for
        now, I did not think how to express convolution with strides via FFT).
        """
        super(PyTorchConv1dAutograd, self).__init__()
        if filter is None:
            if filter_width is None:
                logger.error(
                    "The filter and filter_width cannot be both "
                    "None, provide one of them!")
                sys.exit(1)
            self.filter = Parameter(
                torch.randn(1, 1, filter_width))
        else:
            self.filter = filter
        if bias is None:
            self.bias = Parameter(torch.randn(1))
        else:
            self.bias = bias
        self.padding = padding
        self.index_back = index_back
        self.out_size = out_size
        self.filter_width = filter_width

    def forward(self, input):
        """
        Forward pass of 1D convolution.

        The input consists of N data points with each data point
        representing a signal (e.g., time-series) of length W.

        We also have the notion of channels in the 1-D convolution.
        We want to use more than a single filter even for
        the input signal, so the output is a batch with the same size
        but the number of output channels is equal to the
        number of input filters.

        We want to use the auto-grad (auto-differentiation so call the
        forward method directly).

        :param input: Input data of shape (N, C, W), N - number of data
        points in the batch, C - number of channels, W - the
        width of the signal or time-series (number of data points in
        a univariate series)
        :return: output data, of shape (N, F, W') where W' is given
        by: W' = 1 + (W + 2*pad - WW)

         :see:
         source short: https://goo.gl/GwyhXz
         source full: https://stackoverflow.com/questions/40703751/
         using-fourier-transforms-to-do-convolution?utm_medium=orga
         nic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        >>> # test with compression
        >>> x = np.array([[[1., 2., 3.]]])
        >>> y = np.array([[[2., 1.]]])
        >>> b = np.array([0.0])
        >>> conv_param = {'pad' : 0, 'stride' :1}
        >>> # the 1 index back does not change the result in this case
        >>> expected_result = [4.0, 7.0]
        >>> conv = PyTorchConv1dAutograd(filter=torch.from_numpy(y),
        ... bias=torch.from_numpy(b), index_back=1)
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))

        >>> # test without compression
        >>> x = np.array([[[1., 2., 3.]]])
        >>> y = np.array([[[2., 1.]]])
        >>> b = np.array([0.0])
        >>> conv_param = {'pad' : 0, 'stride' :1}
        >>> dout = np.array([[[0.1, -0.2]]])
        >>> # first, get the expected results from the numpy
        >>> # correlate function
        >>> expected_result = np.correlate(x[0, 0,:], y[0, 0,:],
        ... mode="valid")
        >>> conv = PyTorchConv1dAutograd(filter=torch.from_numpy(y),
        ... bias=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        return PyTorchConv1dFunction.forward(
            ctx=None, input=input, filter=self.filter, bias=self.bias,
            padding=self.padding, index_back=self.index_back,
            out_size=self.out_size)


class PyTorchConv1d(PyTorchConv1dAutograd):
    def __init__(self, filter=None, bias=None, padding=None, index_back=None,
                 out_size=None, filter_width=None):
        super(PyTorchConv1d, self).__init__(
            filter=filter, bias=bias, padding=padding, index_back=index_back,
            out_size=out_size, filter_width=filter_width)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return PyTorchConv1dFunction.apply(input, self.filter,
                                           self.bias,
                                           self.padding,
                                           self.index_back,
                                           self.out_size)


def test_run():
    torch.manual_seed(231)
    filter = np.array([[[1., 2., 3.]]], dtype=np.float32)
    filter = torch.from_numpy(filter)
    module = PyTorchConv1d(filter)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)


if __name__ == "__main__":
    test_run()

    import doctest

    sys.exit(doctest.testmod()[0])
