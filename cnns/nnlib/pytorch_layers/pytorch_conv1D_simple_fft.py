"""
Custom FFT based convolution that can rely on the autograd
(a tape-based automatic differentiation library that supports
all differentiable Tensor operations in pytorch).
"""
import logging
import numpy as np
import torch
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter

from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_signals
from cnns.nnlib.pytorch_layers.pytorch_utils import flip
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


def to_tensor(value):
    """
    Transform from None to -1 or retain the initial value
    for transition as a tensor.
    :param value: a value to be changed to a tensor
    :return: a tensor representing the value, -1 represents None
    """
    if value:
        return tensor([value])
    return tensor([-1])


def from_tensor(tensor_item):
    """
    Transform from tensor to a single numerical value.

    :param tensor_item: tensor with a single value
    :return: a single numerical value extracted from the tensor
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
    def forward(ctx, input, filter, bias, padding=None,
                preserve_energy_rate=None, index_back=None,
                out_size=None):
        """
        Compute the forward pass for the 1D convolution.

        :param ctx: context to save intermediate results
        :param input: the input map to the convolution
        The other parameters are the same as in the
        Conv2dfftAutograd class.
        :return: the result of convolution
        """
        N, C, W = input.size()
        F, C, WW = filter.size()
        fftsize = next_power2(W + WW - 1)
        # pad only the dimensions for the time-series
        # (and neither data points nor the channels)
        padded_x = input
        out_W = W - WW + 1
        if padding is not None:
            padded_x = torch_pad(input, (padding, padding),
                                 'constant', 0)
            out_W += 2 * padding

        if out_size is not None:
            out_W = out_size
            if index_back is not None:
                logger.error(
                    "index_back cannot be set when out_size is used")
                sys.exit(1)
            index_back = len(input) - out_W

        out = torch.zeros([N, F, out_W], dtype=input.dtype,
                          device=input.device)
        for nn in range(N):  # For each time-series in the batch
            for ff in range(F):  # For each filter
                for cc in range(C):  # For each channel (depth)
                    out[nn, ff] += \
                        correlate_signals(
                            padded_x[nn, cc], filter[ff, cc],
                            fftsize, out_size=out_W,
                            preserve_energy_rate=preserve_energy_rate,
                            index_back=index_back)
                    # add the bias term for a given filter
                    out[nn, ff] += bias[ff]

        if ctx:
            ctx.save_for_backward(
                input, filter, bias, to_tensor(padding),
                to_tensor(preserve_energy_rate),
                to_tensor(index_back),
                to_tensor(out_size), to_tensor(fftsize))

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

        :param ctx: context with saved variables
        :param dout: output gradient
        :return: gradients for input map x, filter w and bias b
        """
        x, w, b, padding, preserve_energy_rate, index_back, \
        out_size, fftsize = ctx.saved_tensors
        padding = from_tensor(padding)
        preserve_energy_rate = from_tensor(preserve_energy_rate)
        index_back = from_tensor(index_back)
        out_size = from_tensor(out_size)
        fftsize = from_tensor(fftsize)

        N, C, W = x.shape
        F, C, WW = w.shape
        N, F, W_out = dout.shape

        padded_x = x
        if padding:
            padded_x = torch_pad(input, (padding, padding),
                                 'constant', 0)

        # W = padded_out_W - WW + 1;
        # padded_out_W = W + WW - 1;
        # pad_out = W + WW - 1 // 2  # // 2 - for both sides
        # extend the dout to get the proper final size of dx
        pad_out = (W + WW - 1 - W_out) // 2
        # print("pad_out: ", pad_out)
        if pad_out < 0:
            padded_dout = dout[:, :, abs(pad_out):pad_out]
        else:
            padded_dout = torch_pad(dout, (pad_out, pad_out),
                                    'constant', 0)

        # Initialize gradient output tensors.
        # the x used for convolution was with padding
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        # Calculate dB (the gradient for the bias term).
        # We sum up all the incoming gradients for each filters bias
        # (as in the affine layer).
        for ff in range(F):
            db[ff] += torch.sum(dout[:, ff, :])

        # print("padded x: ", padded_x)
        # print("dout: ", dout)
        # Calculate dw - the gradient for the filters w.
        # By chain rule dw is computed as: dout*x
        # fftsize = next_power2(W + W_out - 1)
        for nn in range(N):
            for ff in range(F):
                for cc in range(C):
                    # accumulate gradient for a filter from each
                    # channel
                    dw[ff, cc] += correlate_signals(
                        padded_x[nn, cc], dout[nn, ff],
                        fftsize, WW,
                        preserve_energy_rate=preserve_energy_rate,
                        index_back=index_back)
                    # print("dw fft: ", dw[ff, cc])

        # Calculate dx - the gradient for the input x.
        # By chain rule dx is dout*w. We need to make dx same shape
        # as padded x for the gradient calculation.
        fftsize = next_power2(padded_dout.shape[-1] + WW - 1)
        for nn in range(N):
            for ff in range(F):
                for cc in range(C):
                    dx[nn, cc] += correlate_signals(
                        padded_dout[nn, ff],
                        flip(w[ff, cc], dim=0), fftsize, W,
                        preserve_energy_rate=preserve_energy_rate,
                        index_back=index_back)

        return dx, dw, db, None, None, None, None


class PyTorchConv1dAutograd(Module):
    def __init__(self, filter=None, bias=None, padding=None,
                 preserve_energy_rate=None, index_back=None,
                 out_size=None, filter_width=None):
        """
        1D convolution using FFT implemented fully in PyTorch.

        :param filter_width: the width of the filter
        :param filter: you can provide the initial filter, i.e.
        filter weights of shape (F, C, WW), where
        F - number of filters, C - number of channels, WW - size of
        the filter
        :param bias: you can provide the initial value of the bias,
        of shape (F,)
        :param padding: the padding added to the front and back of
        the input signal
        :param preserve_energy_rate: the energy of the input to the
        convolution (both, the input map and filter) that
        have to be preserved (the final length is the length of the
        longer signal that preserves the set energy rate).
        :param index_back: how many frequency coefficients should be
        discarded
        :param out_size: what is the expected output size of the
        operation (when compression is used and the out_size is
        smaller than the size of the input to the convolution, then
        the max pooling can be omitted and the compression
        in this layer can serve as the frequency-based (spectral)
        pooling.

        Regarding, the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, it is 1 for the FFT based convolution.
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
        self.preserve_energy_rate = preserve_energy_rate
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

        :param x: Input data of shape (N, C, W), N - number of data
        points in the batch, C - number of channels, W - the
        width of the signal or time-series (number of data points in
        a univariate series)
        :param w:
        :param b: biases
        :param conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive
          fields in the horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad
          the input.
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
        >>> conv_param = {'pad' : 0, 'stride' :1,
        ... 'preserve_energy_rate' :0.9}
        >>> expected_result = [3.5, 7.5]
        >>> conv = Conv2dfftAutograd(filter=torch.from_numpy(y),
        ... bias=torch.from_numpy(b),
        ... preserve_energy_rate=conv_param['preserve_energy_rate'])
        >>> result = conv.forward(x=torch.from_numpy(x))
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
        >>> conv = Conv2dfftAutograd(filter=torch.from_numpy(y),
        ... bias=torch.from_numpy(b))
        >>> result = conv.forward(x=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        return PyTorchConv1dFunction.forward(
            ctx=None, input=input, filter=self.filter,
            bias=self.bias,
            padding=self.padding,
            preserve_energy_rate=self.preserve_energy_rate,
            index_back=self.index_back, out_size=self.out_size)


class PyTorchConv1d(PyTorchConv1dAutograd):
    def __init__(self, filter=None, bias=None, padding=None,
                 preserve_energy_rate=None, index_back=None,
                 out_size=None, filter_width=None):
        super(PyTorchConv1d, self).__init__(
            self, filter=filter, bias=bias, padding=padding,
            preserve_energy_rate=preserve_energy_rate,
            index_back=index_back, out_size=out_size,
            filter_width=filter_width)

    def forward(self, input):
        """
        This is the manual implementation of the forward and
        backward passes via the Function.

        :param input: the input map (image)
        :return: the result of 1D convolution
        """
        return PyTorchConv1dFunction.apply(input, self.filter,
                                           self.bias,
                                           self.padding,
                                           self.preserve_energy_rate,
                                           self.index_back,
                                           self.out_size)


def test_run():
    torch.manual_seed(231)
    module = PyTorchConv1d(3)
    print("filter and bias parameters: ",
          list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)


if __name__ == "__main__":
    # test_run()

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
