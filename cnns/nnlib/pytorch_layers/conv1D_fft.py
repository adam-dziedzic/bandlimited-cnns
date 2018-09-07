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
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class Conv1DfftFunction(torch.autograd.Function):
    """
    Implement the 1D convolution via FFT with compression of the
    input map and the filter.
    """

    @staticmethod
    def forward(ctx, input, filter, bias=None, padding=None, stride=None,
                index_back=None, out_size=None, signal_ndim=1,
                use_next_power2=True, is_manual=tensor([0])):
        """
        Compute the forward pass for the 1D convolution.

        :param ctx: context to save intermediate results, in other
        words, a context object that can be used to stash information
        for backward computation
        :param input: the input map to the convolution (e.g. a time-series).

        The other parameters are similar to the ones in the
        PyTorchConv2dAutograd class.

        :param filter: the filter (a.k.a. kernel of the convolution).
        :param bias: the bias term for each filter.
        :param padding: how much pad each end of the input signal.
        :param stride: what is the stride for the convolution (the pattern for
        omitted values).
        :param index_back: how many last elements in the fft-ed signal to
        discard.
        :param out_size: what is the expected output size - one can disard the
        elements in the frequency domain and do the spectral pooling within the
        convolution.
        :param signal_ndim: this convolution is for 1 dimensional signals.
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param is_manual: to check if the backward computation of convolution
        was computed manually.

        :return: the result of convolution
        """
        print("execute forward pass 1D")
        if index_back is not None and out_size is not None:
            raise TypeError("Specify index_back or out_size not both.")

        # N - number of input maps (or images in the batch),
        # C - number of input channels,
        # W - width of the input (the length of the time-series).
        N, C, W = input.size()

        # F - number of filters,
        # C - number of channels in each filter,
        # WW - the width of the filter (its length).
        F, C, WW = filter.size()

        if padding is None:
            padding_count = 0
        else:
            padding_count = padding

        if out_size is not None:
            out_W = out_size
        else:
            out_W = W - WW + 1  # the length of the output (without padding)
            out_W += 2 * padding_count

        """
        Standard sizes for convolution:
        W - input size
        WW - filter size
        WWW - output size
        
        Forward: [W][WW-1] * [WW]
        Backward dx: [WW-1][WWW][WW-1] * [WW]
        Backward dw: [W][WWW-1] * [WWW]
        
        We can have:
        The required fft-ed map for the dout - the flowing back gradient is of 
        size:
        [WW-1][WWW][WW-1]
        We can add the [WW-1] at the other size of [W]:
        [WW-1][WW][WW-1] - the first forward convolution can be done without any 
        changes, we only have to extract the final output starting from index
        WW-1 (and not from 0). 
        
        
        Sizes for convolution with FFT map reuse:
        WWWW = (WW-1) + WWW + (WW-1)
        """

        init_fft_size_input = W + 2 * padding_count + 2 * (WW - 1)
        if use_next_power2:
            init_fft_size_input = next_power2(init_fft_size_input)

        # The required smallest size for the backward convolution between the
        # flowing back gradient (here its size is represented by out_W) and the
        # filter. This is to compute the gradient for the input map.
        init_fft_size_dout = out_W + 2 * (WW - 1)
        if use_next_power2:
            init_fft_size_dout = next_power2(init_fft_size_dout)

        # The final smallest fft size required.
        init_fft_size = max(init_fft_size_input, init_fft_size_dout)

        # How many padded (zero) values there are because of going to the next
        # power of 2?
        fft_padding_x = init_fft_size - W - 2 * padding_count

        # Pad only the dimensions for the time-series - the width dimension
        # (and neither data points nor the channels).

        padded_x = torch_pad(input,
                             (padding_count, padding_count + fft_padding_x),
                             'constant', 0)

        fft_padding_filter = init_fft_size - WW
        padded_filter = torch_pad(
            filter, (0, fft_padding_filter), 'constant', 0)

        # fft of the input and filters
        xfft = torch.rfft(padded_x, signal_ndim=signal_ndim, onesided=True)
        yfft = torch.rfft(padded_filter, signal_ndim=signal_ndim,
                          onesided=True)

        # The last dimension (-1) has size 2 as it represents the complex
        # numbers with real and imaginary parts. The last but one dimension (-2)
        # represents the length of the signal in the frequency domain.
        init_half_fft_size = xfft.shape[-2]

        half_fft_compressed_size = None
        if index_back is not None:
            half_fft_compressed_size = init_half_fft_size - index_back
        if out_size is not None:
            # We take onesided fft so the output after inverse fft should be out
            # size, thus the representation in spectral domain is twice smaller
            # than the one in time domain.
            half_fft_compressed_size = out_size // 2 + 1

        # Complex numbers are represented as the pair of numbers in the last
        # dimension so we have to narrow the length of the last but one
        # dimension (-2).
        if half_fft_compressed_size is not None:
            xfft = xfft.narrow(dim=-2, start=0, length=half_fft_compressed_size)
            yfft = yfft.narrow(dim=-2, start=0, length=half_fft_compressed_size)

        out = torch.zeros([N, F, out_W], dtype=input.dtype, device=input.device)

        for nn in range(N):  # For each time-series in the batch.
            # Take one time series and unsqueeze it for broadcasting with
            # many filters.
            xfft_nn = xfft[nn].unsqueeze(0)
            out = correlate_fft_signals(
                xfft=xfft_nn, yfft=yfft, fft_size=init_fft_size,
                out_size=out_W)
            out = out[..., :out_W]
            """
            Sum up the elements from computed outputmaps for each input channel.
            Each output map has as many channels as the number of filters. Each
            filter contributes one channel for the output map. 
            """
            out = torch.sum(input=out, dim=1)  # Sum the input channels.
            # `unsqueeze` the dimension for channels.
            out = torch.unsqueeze(input=out, dim=0)
            out[nn] = out
            if bias is not None:
                # Add the bias term for each filter.
                # Bias has to be unsqueezed to the dimension of the out to
                # properly sum up the values.
                out[nn] += bias.unsqueeze(1)

        # TODO: how to compute the backward pass for the strided FFT convolution?
        # Add additional zeros in the places of the output that were removed
        # by striding.
        if stride is not None and stride > 1:
            out = out[:, :, 0::stride]

        if ctx:
            ctx.save_for_backward(xfft, yfft, to_tensor(W), to_tensor(WW),
                                  to_tensor(init_fft_size), is_manual)

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
        print("execute backward pass 1D")
        xfft, yfft, W, WW, init_fft_size, is_manual = ctx.saved_tensors
        is_manual[0] = 1  # Mark the manual execution of the backward pass.
        W = from_tensor(W)
        WW = from_tensor(WW)
        init_fft_size = from_tensor(init_fft_size)
        signal_ndim = 1

        # The last dimension (_) for xfft and yfft is the 2 element complex
        # number.
        N, C, half_fft_compressed_size, _ = xfft.shape
        F, C, half_fft_compressed_size, _ = yfft.shape
        N, F, out_W = dout.shape

        dx = dw = db = None

        # Take the fft of dout (the gradient of the output of the forward pass).
        # We have to pad the flowing back gradient in the time (spatial) domain,
        # since it does not give correct results even for the case without
        # compression if we pad in the spectral (frequency) domain.
        # We pad both sides of the dout gradient. The left side is padded by
        # (WW-1), the right side is padded also by (WW-1) and the additional
        # zeros that are required to fill in the init_fft_size.
        left_pad = right_pad = WW - 1
        fft_pad = init_fft_size - (left_pad + out_W + right_pad)
        dout = torch_pad(dout, (left_pad, right_pad + fft_pad), 'constant', 0)
        doutfft = torch.rfft(dout, signal_ndim=signal_ndim, onesided=True)

        # If the compression was done in the forward pass, then we have to
        # compress the pure fft-ed version of the flowing back gradient:
        # doutftt.
        init_half_fft_size = doutfft.shape[-2]
        if half_fft_compressed_size < init_half_fft_size:
            doutfft = doutfft.narrow(dim=-2, start=0,
                                     length=half_fft_compressed_size)

        if ctx.needs_input_grad[0]:
            # Initialize gradient output tensors.
            # the x used for convolution was with padding
            """
            More specifically:
            if the forward convolution is: [x1, x2, x3, x4] * [w1, w2], where *
            denotes the convolution operation, 
            Conv (out) = [x1 w1 + x2 w2, x2 w1 + x3 w2, x3 w1 + x4 w2]
            then the backward pass to compute the gradient for the inputs is 
            as follows (L - is the Loss function):
            gradient L / gradient x = 
            gradient L / gradient Conv (times) gradient Conv / gradient x =
            dout x gradient Conv / gradient x = (^)

            gradient Conv / gradient x1 = [w1, 0, 0]
            gradient Conv / gradient x2 = [w2, w1, 0]
            gradient Conv / gradient x3 = [0, w2, w1]
            gradient Conv / gradient x4 = [0, 0, w2]

            dout = [dx1, dx2, dx3]

            gradient L / gradient x1 = dout * gradient Conv / gradient x1 =
            [dx1*w1 + dx2*0 + dx3*0] = [dx1*w1]

            gradient L / gradient x2 = dout * gradient Conv / gradient x2 =
            [dx1*w2 + dx2*w1 + dx3*0] = [dx1*w2 + dx2*w1]
            
            gradient L / gradient x3 = dout * gradient Conv / gradient x3 =
            [dx1*0 + dx2*w2 + dx3*w1] = [dx2*w2 + dx3*w1]
            
            gradient L / gradient x4 = dout * gradient Conv / gradient x3 =
            [dx1*0 + dx2*0 + dx3*w2] = [dx3*w2]

            Thus, the gradient for the weights is the convolution between the 
            flowing back gradient dout and the input x:
            # The flowing back gradient is padded from both sides, the padding
            # size is: WW-1 (roughly the size of the filter)
            gradient L / gradient w = [0, dx1, dx2, dx3, 0] * flip([w1, w2]) = 
            [0, dx1, dx2, dx3, 0] * [w2, w1] = 
            [dx1*w1, dx1*w2 + dx2*w1, dx2*w2 + dx3*w1, dx3*w2]
            """
            dx = torch.zeros([N, C, W], dtype=xfft.dtype)
            conjugate_yfft = pytorch_conjugate(yfft)
            for nn in range(N):
                # Take one time series and unsqueeze it for broadcast with
                # many gradients dout.
                doutfft_nn = doutfft[nn, :, :].unsqueeze(0)
                out = correlate_fft_signals(
                    xfft=doutfft_nn, yfft=conjugate_yfft,
                    fft_size=init_fft_size, out_size=W)
                out = out[:, :, left_pad: left_pad + out_W]
                out = torch.sum(out, dim=1)
                out = torch.unsqueeze(input=out, dim=1)
                dx[nn] = out

        if ctx.needs_input_grad[1]:
            dw = torch.zeros([F, C, WW], dtype=yfft.dtype)
            # Calculate dw - the gradient for the filters w.
            # By chain rule dw is computed as: dout*x
            """
            More specifically:
            if the forward convolution is: [x1, x2, x3, x4] * [w1, w2], where *
            denotes the convolution operation, 
            Conv (out) = [x1 w1 + x2 w2, x2 w1 + x3 w2, x3 w1 + x4 w2]
            then the backward pass to compute the gradient for the weights is 
            as follows (L - is the Loss function):
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
                # Gather all the contributions to the output that were caused
                # by a given filter.
                doutfft_ff = doutfft[:, ff, :].unsqueeze(1)
                out = correlate_fft_signals(
                    xfft=xfft, yfft=doutfft_ff, fft_size=init_fft_size,
                    out_size=WW, signal_ndim=signal_ndim, is_forward=False)
                # For a given filter, we have to sum up all its contributions
                # to all the input maps.
                out = torch.sum(input=out, dim=0)
                # `unsqueeze` the dimension 0 for the input data points (N).
                out = torch.unsqueeze(input=out, dim=0)
                dw[ff] = out

        if ctx.needs_input_grad[2]:
            # The number of bias elements is equal to the number of filters.
            db = torch.zeros(F)

            # Calculate dB (the gradient for the bias term).
            # We sum up all the incoming gradients for each filter
            # bias (as in the affine layer).
            for ff in range(F):
                db[ff] += torch.sum(dout[:, ff, :])

        return dx, dw, db, None, None, None, None, None, None, None


class Conv1DfftAutograd(Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 index_back=None, out_size=None, filter_value=None,
                 bias_value=None, use_next_power2=True,
                 is_manual=tensor([0])):
        """
        1D convolution using FFT implemented fully in PyTorch.

        :param in_channels: (int) – Number of channels in the input series.
        :param out_channels: (int) – Number of channels produced by the
        convolution.
        :param kernel_size: (int) - Size of the convolving kernel (the width of
        the filter).
        :param stride: what is the stride for the convolution (the pattern for
        omitted values).
        :param padding: the padding added to the front and back of
        the input signal.
        :param dilation: (int) – Spacing between kernel elements. Default: 1
        :param groups: (int) – Number of blocked connections from input channels
        to output channels. Default: 1
        :param bias: (bool) - add bias or not
        :param index_back: how many frequency coefficients should be
        discarded
        :param out_size: what is the expected output size of the
        operation (when compression is used and the out_size is
        smaller than the size of the input to the convolution, then
        the max pooling can be omitted and the compression
        in this layer can serve as the frequency-based (spectral)
        pooling.
        :param filter_value: you can provide the initial filter, i.e.
        filter weights of shape (F, C, WW), where
        F - number of filters, C - number of channels, WW - size of
        the filter
        :param bias_value: you can provide the initial value of the bias,
        of shape (F,)
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param is_manual: to check if the backward computation of convolution
        was computed manually.

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, we can generate the full output, and then remove the
        redundant elements according to the stride parameter. We have to figure
        out how to run the backward pass for this strided FFT-based convolution.
        """
        super(Conv1DfftAutograd, self).__init__()
        if dilation > 1:
            raise NotImplementedError("dilation > 1 is not supported.")
        if groups > 1:
            raise NotImplementedError("groups > 1 is not supported.")

        if filter_value is None:
            if out_channels is None or in_channels is None or \
                    kernel_size is None:
                raise ValueError("Either specify filter_value or provide all"
                                 "the required parameters (out_channels, "
                                 "in_channels and kernel_size) to generate the "
                                 "filter.")
            self.filter = Parameter(
                torch.randn(out_channels, in_channels, kernel_size))
        else:
            self.filter = filter_value
        if bias_value is None:
            if bias is True:
                self.bias = Parameter(torch.randn(out_channels, 1))
            else:
                self.bias = None
        else:
            self.bias = bias_value
        self.padding = padding
        self.index_back = index_back
        self.out_size = out_size
        self.filter_width = kernel_size
        self.use_next_power2 = use_next_power2
        self.stride = stride
        self.signal_ndim=1
        self.is_manual = is_manual

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
        >>> expected_result = [3.5, 7.5]
        >>> conv = Conv1DfftAutograd(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b), index_back=1)
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
        >>> conv = Conv1DfftAutograd(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        return Conv1DfftFunction.forward(
            ctx=None, input=input, filter=self.filter, bias=self.bias,
            padding=self.padding, stride=self.stride,
            index_back=self.index_back, out_size=self.out_size,
            use_next_power2=self.use_next_power2, is_manual=self.is_manual)


class Conv1Dfft(Conv1DfftAutograd):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=None, padding=None, bias=None, index_back=None,
                 out_size=None, filter_value=None, bias_value=None):
        super(Conv1Dfft, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            index_back=index_back, out_size=out_size, filter_value=filter_value,
            bias_value=bias_value)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return Conv1DfftFunction.apply(
            input, self.filter, self.bias, self.padding, self.stride,
            self.index_back, self.out_size, self.signal_ndim,
            self.use_next_power2, self.is_manual)


def test_run():
    torch.manual_seed(231)
    filter = np.array([[[1., 2., 3.]]], dtype=np.float32)
    filter = torch.from_numpy(filter)
    module = Conv1Dfft(filter_value=filter)
    print()
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)
    assert module.is_manual[0] == 1
    print("The manual backprop was executed.")


if __name__ == "__main__":
    test_run()

    import doctest
    sys.exit(doctest.testmod()[0])