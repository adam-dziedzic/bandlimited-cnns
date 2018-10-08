"""
Custom FFT based convolution that:
1) computes forward and backward manually
2) manually computes the forward pass and relies on the autograd (a tape-based
automatic differentiation library that supports all differentiable Tensor
operations in PyTorch) for automatic differentiation for the backward pass.
"""
import logging
import sys

import numpy as np
import torch
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter

from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_fft_signals2D
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import get_pair
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class PyTorchConv2dFunction(torch.autograd.Function):
    """
    Implement the 2D convolution via FFT with compression in the spectral domain
    of the input map and the filter.
    """

    @staticmethod
    def forward(ctx, input, filter, bias, padding=None, stride=None,
                index_back=None, out_size=None, use_next_power2=True):
        """
        Compute the forward pass for the 2D convolution.

        :param ctx: context to save intermediate results, in other words,
        a context object that can be used to stash information for backward
        computation.
        :param input: the input map to the convolution (e.g. a time-series).

        The other parameters are similar to the ones in the
        PyTorchConv2dAutograd class.

        :param filter: the filter (a.k.a. kernel of the convolution).
        :param bias: the bias term for each filter.
        :param padding: how much to pad each end of the height and width of the
        input map, implicit applies zero padding on both sides of the input. It
        can be a single number or a tuple (padH, padW).
        Default: None (no padding).
        :param stride: what is the stride for the height and width dimensions
        when convolving the input map with the filter, implicitly we do not
        apply the stride (move one pixel at a time).
        Default: None (no padding).
        :param index_back: how many of the last height and width elements in the
        fft-ed map to discard. It Can be a single number or a tuple
        (index_back_H, index_back_W). Default: None (no compression).
        :param out_size: what is the expected output size - one can discard
        the elements in the frequency domain and do the spectral pooling within
        the convolution. It can be a single number or a tuple (outH, outW).
        Default: None (the standard size, e.g., outW = W - WW + 1).

        :return: the result of convolution.
        """
        print("execute forward pass")
        if index_back is not None and out_size is not None:
            raise TypeError("Specify either index_back or out_size not both.")

        signal_ndim = 2
        # N - number of input maps (or images in the batch).
        # C - number of input channels.
        # H - height of the input map (e.g., height of an image).
        # W - width of the input map (e.g. width of an image).
        N, C, H, W = input.size()

        # F - number of filters.
        # C - number of channels in each filter.
        # HH - the height of the filter.
        # WW - the width of the filter (its length).
        F, C, HH, WW = filter.size()

        index_back_H, index_back_W = get_pair(value=index_back, val_1_default=0,
                                              val2_default=0)

        pad_H, pad_W = get_pair(value=padding, val_1_default=0, val2_default=0,
                                name="padding")

        out_size_H, out_size_W = get_pair(value=out_size, val_1_default=None,
                                          val2_default=None, name="out_size")

        stride_H, stride_W = get_pair(value=stride, val_1_default=None,
                                      val2_default=None, name="strideŁ")

        if out_size_H is not None:
            out_H = out_size_H
        else:
            out_H = H - HH + 1  # the height of the output (without padding)
            out_H += 2 * pad_H

        if out_size_W is not None:
            out_W = out_size_W
        else:
            out_W = W - WW + 1  # the width of the output (without padding)
            out_W += 2 * pad_W

        # We have to pad input with (WW - 1) to execute fft correctly (no
        # overlapping signals) and optimize it by extending the signal to the
        # next power of 2. We want to reuse the fft-ed input x, so we use the
        # larger size chosen from: the filter width WW or output width out_W.
        # Larger padding does not hurt correctness of fft but makes it slightly
        # slower, in terms of the computation time.
        HHH = max(out_H, HH)
        init_fft_H = H + 2 * pad_H + HHH - 1

        WWW = max(out_W, WW)
        init_fft_W = W + 2 * pad_W + WWW - 1

        if use_next_power2 is True:
            init_fft_H = next_power2(init_fft_H)
            init_fft_W = next_power2(init_fft_W)

        # How many padded (zero) values there are because of going to the next
        # power of 2?
        fft_padding_input_H = init_fft_H - 2 * pad_H - H
        fft_padding_input_W = init_fft_W - 2 * pad_W - W

        # Pad only the dimensions for the height and width and neither the data
        # points (the batch dimension) nor the channels.
        padded_x = torch_pad(
            input, (pad_W, pad_W + fft_padding_input_W, pad_H,
                    pad_H + fft_padding_input_H), 'constant', 0)

        fft_padding_filter_H = init_fft_H - HH
        fft_padding_filter_W = init_fft_W - WW

        padded_filter = torch_pad(
            filter, (0, fft_padding_filter_W, 0, fft_padding_filter_H),
            'constant', 0)

        # fft of the input and filters
        xfft = torch.rfft(padded_x, signal_ndim=signal_ndim, onesided=True)
        yfft = torch.rfft(padded_filter, signal_ndim=signal_ndim,
                          onesided=True)

        # The last dimension (-1) has size 2 as it represents the complex
        # numbers with real and imaginary parts. The last but one dimension (-2)
        # represents the length of the signal in the frequency domain.
        init_half_fft_W = xfft.shape[-2]
        init_half_fft_H = xfft.shape[-3]

        # Compute how much to compress the fft-ed signal for its width (W).
        half_fft_W = init_half_fft_W
        if index_back_W is not None:
            half_fft_W = init_half_fft_W - index_back_W
        if out_size_W is not None:
            # We take one-sided fft so the output after the inverse fft should
            # be out size, thus the representation in the spectral domain is
            # twice smaller than the one in the time domain.
            half_fft_W = out_size_W // 2 + 1

        # Complex numbers are represented as the pair of numbers in the last
        # dimension so we have to narrow the length of the last but one
        # dimension.
        if half_fft_W < init_half_fft_W:
            xfft = xfft.narrow(dim=-2, start=0, length=half_fft_W)
            yfft = yfft.narrow(dim=-2, start=0, length=half_fft_W)

        # Compute how much to compress the fft-ed signal for its height (H).
        half_fft_H = init_half_fft_H
        if index_back_H is not None:
            half_fft_H = init_half_fft_H - index_back_H
        if out_size_H is not None:
            # We take one-sided fft so the output after the inverse fft should
            # be out size, thus the representation in the spectral domain is
            # twice smaller than the one in time domain.
            half_fft_H = out_size_H // 2 + 1

        # Complex numbers are represented as the pair of numbers in the last
        # dimension so we have to narrow the length of the last but one
        # dimension.
        if half_fft_H < init_half_fft_H:
            xfft = xfft.narrow(dim=-3, start=0, length=half_fft_H)
            yfft = yfft.narrow(dim=-3, start=0, length=half_fft_H)

        out = torch.zeros([N, F, out_H, out_W], dtype=input.dtype,
                          device=input.device)

        for nn in range(N):  # For each time-series in the batch.
            # Take one time series and unsqueeze it for broadcasting with
            # many filters.
            xfft_nn = xfft[nn].unsqueeze(0)
            out[nn] = correlate_fft_signals2D(
                xfft=xfft_nn, yfft=yfft,
                input_height=init_fft_H, input_width=init_fft_W,
                half_fft_height=init_half_fft_H, half_fft_width=init_half_fft_W,
                out_height=out_H, out_width=out_W,
                is_forward=True)
            if bias is not None:
                # Add the bias term for each filtekr (it has to be unsqueezed to
                # the dimension of the out to properly sum up the values).
                out[nn] += bias.unsqueeze(-1).unsqueeze(-1)

        # TODO: how to compute the backward pass for the strided FFT convolution
        if stride_W is not None or stride_H is not None:
            if stride_W is None:
                stride_W = 1
            if stride_H is None:
                stride_H = 1
            if stride_H != 1 or stride_W != 1:
                out = out[:, :, 0::stride_H, 0::stride_W]

        if ctx:
            ctx.save_for_backward(xfft, yfft, to_tensor(H), to_tensor(HH),
                                  to_tensor(W), to_tensor(WW),
                                  to_tensor(init_fft_H), to_tensor(init_fft_W))

        return out

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
        # logger.debug("execute backward")
        print("execute backward pass")
        xfft, yfft, H, HH, W, WW, init_fft_H, init_fft_W = ctx.saved_tensors
        H = from_tensor(H)
        HH = from_tensor(HH)
        W = from_tensor(W)
        WW = from_tensor(WW)
        init_fft_H = from_tensor(init_fft_H)
        init_fft_W = from_tensor(init_fft_W)
        signal_ndim = 2

        # The last dimension for xfft and yfft is the 2 element complex number.
        N, C, half_fft_compressed_H, half_fft_compressed_W, _ = xfft.shape
        F, C, half_fft_compressed_H, half_fft_compressed_W, _ = yfft.shape
        N, F, H_out, W_out = dout.shape

        dx = dw = db = None

        # Take the fft of dout (the gradient of the output of the forward pass).
        # We have to pad the flowing back gradient in the time (spatial) domain,
        # since it does not give correct results even for the case without
        # compression if we pad in the spectral (frequency) domain.
        fft_padding_dout_H = init_fft_H - H_out
        fft_padding_dout_W = init_fft_W - W_out

        padded_dout = torch_pad(
            dout, (0, fft_padding_dout_W, 0, fft_padding_dout_H),
            'constant', 0)

        doutfft = torch.rfft(padded_dout, signal_ndim=signal_ndim,
                             onesided=True)

        # the last dimension is for real and imaginary parts of the complex
        # numbers
        N, F, init_half_fft_H, init_half_fft_W, _ = doutfft.shape

        if half_fft_compressed_H < init_half_fft_H:
            doutfft = doutfft.narrow(dim=-3, start=0,
                                     length=half_fft_compressed_H)
        if half_fft_compressed_W < init_half_fft_W:
            doutfft = doutfft.narrow(dim=-2, start=0,
                                     length=half_fft_compressed_W)

        if ctx.needs_input_grad[0]:
            # Initialize gradient output tensors.
            # the x used for convolution was with padding
            dx = torch.zeros([N, C, H, W], dtype=xfft.dtype)
            conjugate_yfft = pytorch_conjugate(yfft)
            for nn in range(N):
                # Take one time series and unsqueeze it for broadcast with
                # many gradients dout.
                doutfft_nn = doutfft[nn].unsqueeze(0)
                dx[nn] = correlate_fft_signals2D(
                    xfft=doutfft_nn, yfft=conjugate_yfft,
                    input_height=init_fft_H, input_width=init_fft_W,
                    half_fft_height=init_half_fft_H,
                    half_fft_width=init_half_fft_W,
                    out_height=H, out_width=W,
                    is_forward=False)

        if ctx.needs_input_grad[1]:
            dw = torch.zeros([F, C, HH, WW], dtype=yfft.dtype)
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
                dw[ff] = correlate_fft_signals2D(
                    xfft=xfft, yfft=doutfft_ff,
                    input_height=init_fft_H, input_width=init_fft_W,
                    half_fft_height=init_half_fft_H,
                    half_fft_width=init_half_fft_W,
                    out_height=HH, out_width=WW,
                    is_forward=False)

        if ctx.needs_input_grad[2]:
            # The number of bias elements is equal to the number of filters.
            db = torch.zeros(F)

            # Calculate dB (the gradient for the bias term).
            # We sum up all the incoming gradients for each filter
            # bias (as in the affine layer).
            for ff in range(F):
                db[ff] += torch.sum(dout[:, ff, :])

        return dx, dw, db, None, None, None, None, None


class PyTorchConv2dAutograd(Module):
    def __init__(self, filter_value=None, bias=None, padding=None, stride=None,
                 index_back=None, out_size=None, filter_width=None,
                 use_next_power2=True):
        """
        2D convolution using FFT implemented fully in PyTorch.

        :param filter_value: you can provide the initial filter, i.e.
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
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param stride: what is the stride for the convolution (the pattern for
        omitted values).

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, it has to be 1 for the FFT based convolution (at least for
        now, I did not think how to express convolution with strides via FFT).
        """
        super(PyTorchConv2dAutograd, self).__init__()
        if filter_value is None:
            if filter_width is None:
                logger.error(
                    "The filter and filter_width cannot be both "
                    "None, provide one of them!")
                sys.exit(1)
            self.filter = Parameter(
                torch.randn(1, 1, filter_width))
        else:
            self.filter = filter_value
        if bias is None:
            self.bias = Parameter(torch.randn(1))
        else:
            self.bias = bias
        self.padding = padding
        self.stride = stride
        self.index_back = index_back
        self.out_size = out_size
        self.filter_width = filter_width
        self.use_next_power2 = use_next_power2

    def forward(self, input):
        """
        Forward pass of 2D convolution.

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

        >>> # Test stride for 3 channels and 2 filters.
        >>> # based on: http://cs231n.github.io/convolutional-networks/
        >>> x = tensor(
        ... [[[
        ... [2.0, 0.0, 2.0, 0.0, 1.0],
        ... [0.0, 1.0, 0.0, 1.0, 2.0],
        ... [0.0, 2.0, 0.0, 0.0, 0.0],
        ... [2.0, 2.0, 2.0, 0.0, 2.0],
        ... [2.0, 0.0, 1.0, 1.0, 1.0],
        ... ],[
        ... [1.0, 1.0, 2.0, 1.0, 0.0],
        ... [0.0, 1.0, 2.0, 2.0, 1.0],
        ... [0.0, 1.0, 2.0, 1.0, 2.0],
        ... [2.0, 1.0, 0.0, 2.0, 1.0],
        ... [2.0, 1.0, 2.0, 1.0, 2.0],
        ... ],[
        ... [1.0, 1.0, 2.0, 1.0, 2.0],
        ... [1.0, 2.0, 0.0, 0.0, 1.0],
        ... [0.0, 0.0, 2.0, 0.0, 0.0],
        ... [1.0, 2.0, 2.0, 0.0, 2.0],
        ... [0.0, 2.0, 2.0, 0.0, 0.0],
        ... ]]])
        >>> y = tensor([[
        ... [[ 1.0, 0.0, 0.0],
        ...  [-1.0,-1.0, 0.0],
        ...  [ 1.0, 0.0, 0.0]],
        ... [[-1.0, 0.0, 1.0],
        ...  [ 0.0, 1.0,-1.0],
        ...  [-1.0,-1.0, 1.0]],
        ... [[-1.0, 0.0, 1.0],
        ...  [ 0.0, 0.0, 1.0],
        ...  [ 0.0,-1.0,-1.0]]],
        ... [
        ... [[ 1.0, 0.0,-1.0],
        ...  [-1.0, 0.0, 1.0],
        ...  [-1.0, 1.0, 0.0]],
        ... [[-1.0,-1.0, 0.0],
        ...  [ 0.0, -1.0, 1.0],
        ...  [ 1.0, 1.0,-1.0]],
        ... [[ 1.0,-1.0, 1.0],
        ...  [ 0.0,-1.0,-1.0],
        ...  [ 1.0, 1.0,-1.0]]],
        ... ])
        >>> b = tensor([1.0, 0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0,
        ... padding=(1, 1), stride=(2, 2))
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[
        ... [-2.0, 1.0,-3.0],
        ... [-1.0, 1.0,-3.0],
        ... [ 5.0, 2.0,-1.0]],[
        ... [-4.0,-2.0, 3.0],
        ... [ 5.0,-3.0, 2.0],
        ... [-6.0,-1.0,-8.0]],
        ... ]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result, decimal=5,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test 3 channels and 2 filters.
        >>> x = tensor(
        ... [[[
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ... [0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0],
        ... [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0],
        ... [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ... [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0],
        ... [0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ... ],[
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ... [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0],
        ... [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
        ... [0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        ... [0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
        ... [0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ... ],[
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ... [0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        ... [0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        ... [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        ... [0.0, 1.0, 2.0, 2.0, 0.0, 2.0, 0.0],
        ... [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        ... [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ... ]]])
        >>> y = tensor([[
        ... [[ 1.0, 0.0, 0.0],
        ...  [-1.0,-1.0, 0.0],
        ...  [ 1.0, 0.0, 0.0]],
        ... [[-1.0, 0.0, 1.0],
        ...  [ 0.0, 1.0,-1.0],
        ...  [-1.0,-1.0, 1.0]],
        ... [[-1.0, 0.0, 1.0],
        ...  [ 0.0, 0.0, 1.0],
        ...  [ 0.0,-1.0,-1.0]]],
        ... [
        ... [[ 1.0, 0.0,-1.0],
        ...  [-1.0, 0.0, 1.0],
        ...  [-1.0, 1.0, 0.0]],
        ... [[-1.0,-1.0, 0.0],
        ...  [ 0.0, -1.0, 1.0],
        ...  [ 1.0, 1.0,-1.0]],
        ... [[ 1.0,-1.0, 1.0],
        ...  [ 0.0,-1.0,-1.0],
        ...  [ 1.0, 1.0,-1.0]]],
        ... ])
        >>> b = tensor([1.0, 0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0,
        ... padding=0)
        >>> result = conv.forward(x=x)
        >>> expect = np.array(
        ... [[[[-2.0000e+00, -1.0000e+00,  1.0000e+00, -2.0000e+00, -3.0000e+00],
        ... [ 5.0000e+00,  2.0000e+00, -2.0000e+00,  1.0000e+00, -6.0000e+00],
        ... [-1.0000e+00, -4.0000e+00,  1.0000e+00, -1.0000e+00, -3.0000e+00],
        ... [ 1.7881e-07,  1.0000e+00, -7.0000e+00, -4.7684e-07, -3.0000e+00],
        ... [ 5.0000e+00,  1.0000e+00,  2.0000e+00,  1.0000e+00, -1.0000e+00]],
        ... [[-4.0000e+00,  1.0000e+00, -2.0000e+00, -2.0000e+00,  3.0000e+00],
        ... [-3.0000e+00, -2.0000e+00, -1.0000e+00,  4.0000e+00, -2.0000e+00],
        ... [ 5.0000e+00,  1.0000e+00, -3.0000e+00, -5.0000e+00,  2.0000e+00],
        ... [-3.0000e+00, -5.0000e+00,  2.0000e+00, -1.0000e+00, -3.0000e+00],
        ... [-6.0000e+00, -6.0000e+00, -1.0000e+00,  3.0000e+00, -8.0000e+00]]]]
        ... )
        >>> np.testing.assert_array_almost_equal(x=expect, y=result, decimal=5,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test with compression (index back)
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=1,
        ... use_next_power2=False)
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test 2D convolution.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0)
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test bias.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([-1.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0)
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[[21.0, 21.0], [17., 13.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Don't use next power of 2.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0,
        ... use_next_power2=False)
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test 2 channels and 2 filters.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
        ... [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]])
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0],[3.0, -2.0]]],
        ... [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]])
        >>> b = tensor([0.0, 0.0])
        >>> conv = PyTorchConv2dAutograd(filter_value=y, bias=b, index_back=0)
        >>> result = conv.forward(x=x)
        >>> expect = np.array([[[[23.0, 32.0], [30., 4.]],[[11.0, 12.0],
        ... [13.0, -11.0]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result, decimal=5,
        ... err_msg="The expected array x and computed y are not almost equal.")
        """
        return PyTorchConv2dFunction.forward(
            ctx=None, input=input, filter=self.filter, bias=self.bias,
            padding=self.padding, stride=self.stride,
            index_back=self.index_back, out_size=self.out_size,
            use_next_power2=self.use_next_power2)


class PyTorchConv2d(PyTorchConv2dAutograd):
    def __init__(self, filter_value=None, bias=None, padding=None, stride=None,
                 index_back=None, out_size=None, filter_width=None,
                 use_next_power2=False):
        super(PyTorchConv2d, self).__init__(
            filter_value=filter_value, bias=bias, padding=padding, stride=stride,
            index_back=index_back, out_size=out_size, filter_width=filter_width,
            use_next_power2=use_next_power2)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 2D convolution
        """
        return PyTorchConv2dFunction.apply(input, self.filter,
                                           self.bias,
                                           self.padding,
                                           self.stride,
                                           self.index_back,
                                           self.out_size,
                                           self.use_next_power2)


def test_run():
    torch.manual_seed(231)
    filter = np.array([[[[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [0.0, 1.0, -1.0]]]],
                      dtype=np.float32)
    filter = torch.from_numpy(filter)
    module = PyTorchConv2d(filter)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8, 8))
    print("gradient for the input: ", input.grad)


if __name__ == "__main__":
    test_run()

    import doctest

    sys.exit(doctest.testmod()[0])
