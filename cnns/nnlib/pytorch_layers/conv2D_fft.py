"""
Custom FFT based convolution that:
1) computes forward and backward manually (this is the main part);
2) manually computes the forward pass and relies on the autograd (a tape-based
automatic differentiation library that supports all differentiable Tensor
operations in PyTorch) for automatic differentiation for the backward pass.
"""
import logging
import sys
import math
import numpy as np
import torch
import time
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter

from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_fft_signals2D
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import get_pair
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2
from cnns.nnlib.pytorch_layers.pytorch_utils import preserve_energy2D_symmetry
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_odd
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_odd_index_back
from cnns.nnlib.pytorch_layers.pytorch_utils import zero_out_min
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_elem_size
from cnns.nnlib.pytorch_layers.pytorch_utils import get_tensors_elem_size
from cnns.nnlib.pytorch_layers.pytorch_utils import get_step_estimate
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.arguments import Arguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class Conv2dfftFunction(torch.autograd.Function):
    """
    Implement the 2D convolution via FFT with compression in the spectral domain
    of the input map (activation) and the filter.
    """
    signal_ndim = 2

    @staticmethod
    def forward(ctx, input, filter, bias=None, padding=(0, 0), stride=(1, 1),
                args=None, out_size=None, is_manual=tensor([0]),
                conv_index=None):
        """
        Compute the forward pass for the 2D convolution.

        :param ctx: context to save intermediate results, in other words,
        a context object that can be used to stash information for backward
        computation.
        :param input: the input map (activation) to the convolution (e.g. an
        image).
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
        :param preserve_energy: how much energy of the input images should be
        preserved.
        :param out_size: what is the expected output size - one can discard
        the elements in the frequency domain and do the spectral pooling within
        the convolution. It can be a single number or a tuple (outH, outW). It
        is more flexible than the pooling or striding.
        Default: None (the standard size, e.g., outW = W - WW + 1).
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param is_manual: to check if the backward computation of convolution
        was computed manually.
        :param conv_index: the index of the convolution.
        :param is_debug: is the debug mode of execution.
        :param compress_type: NO_FILTER - should the filter be compressed or
        only the input signal? BIG_COEF: should we keep only the largest
        coefficients or delete the coefficients from the end of the signal
        representation in the frequency domain? STANDARD: cut off the same
        number of coefficients for each signal and filter in the batch based on
        the whole energy of the signals in the batch.

        :return: the result of convolution.
        """
        print("input size: ", input.size())
        if args is not None:
            index_back = args.index_back
            preserve_energy = args.preserve_energy
            use_next_power2 = args.next_power2
            is_debug = args.is_debug
            compress_type = args.compress_type
            stride_type = args.stride_type
        else:
            index_back = None
            preserve_energy = None
            use_next_power2 = False
            is_debug = False
            compress_type = CompressType.STANDARD
            stride_type = StrideType.STANDARD

        dtype = input.dtype
        device = input.device

        if is_debug:
            pass
            # print("execute forward pass")
            # torch.set_printoptions(threshold=5000)
            # print("input 0: ", input[0])
            # print("filter 0: ", filter[0])

        INPUT_ERROR = "Specify only one of: index_back, out_size, or " \
                      "preserve_energy"
        if (index_back is not None and index_back > 0) and out_size is not None:
            raise TypeError(INPUT_ERROR)
        if (index_back is not None and index_back > 0) and (
                preserve_energy is not None and preserve_energy < 100):
            raise TypeError(INPUT_ERROR)
        if out_size is not None and (
                preserve_energy is not None and preserve_energy < 100):
            raise TypeError(INPUT_ERROR)

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

        index_back_H, index_back_W = get_pair(value=index_back,
                                              val_1_default=None,
                                              val2_default=None)
        if index_back_H != index_back_W:
            raise Exception(
                "We only support a symmetric compression in the frequency domain.")

        pad_H, pad_W = get_pair(value=padding, val_1_default=0, val2_default=0,
                                name="padding")

        if pad_H != pad_W:
            raise Exception(
                "We only support a symmetric padding in the frequency domain.")

        out_size_H, out_size_W = get_pair(value=out_size, val_1_default=None,
                                          val2_default=None, name="out_size")

        if out_size_H != out_size_W:
            raise Exception(
                "We only support a symmetric outputs in the frequency domain.")

        stride_H, stride_W = get_pair(value=stride, val_1_default=None,
                                      val2_default=None, name="stride")

        if stride_H != stride_W:
            raise Exception(
                "We only support a symmetric striding in the frequency domain.")

        if out_size_H:
            out_H = out_size_H
        elif out_size or stride_type is StrideType.SPECTRAL:
            out_H = (H - HH + 2 * pad_H) // stride_H + 1
        else:
            out_H = H - HH + 1 + 2 * pad_H

        if out_size_W:
            out_W = out_size_W
        elif out_size or stride_type is StrideType.SPECTRAL:
            out_W = (W - WW + 2 * pad_W) // stride_W + 1
        else:
            out_W = W - WW + 1 + 2 * pad_W

        if out_H != out_W:
            raise Exception(
                "We only support a symmetric compression in the frequency domain.")

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
        del input

        fft_padding_filter_H = init_fft_H - HH
        fft_padding_filter_W = init_fft_W - WW

        padded_filter = torch_pad(
            filter, (0, fft_padding_filter_W, 0, fft_padding_filter_H),
            'constant', 0)
        del filter

        # fft of the input and filters
        xfft = torch.rfft(padded_x, signal_ndim=Conv2dfftFunction.signal_ndim,
                          onesided=True)
        del padded_x

        yfft = torch.rfft(padded_filter,
                          signal_ndim=Conv2dfftFunction.signal_ndim,
                          onesided=True)
        del padded_filter

        # The last dimension (-1) has size 2 as it represents the complex
        # numbers with real and imaginary parts. The last but one dimension (-2)
        # represents the length of the signal in the frequency domain.
        init_half_fft_W = xfft.shape[-2]
        init_fft_H = xfft.shape[-3]

        # Pooling either via stride or explicitly via out_size_W.
        if out_size or stride_type is StrideType.SPECTRAL:
            # We take one-sided fft so the output after the inverse fft should
            # be out size, thus the representation in the spectral domain is
            # twice smaller than the one in the spatial domain.
            half_fft_W = out_W // 2 + 1
            xfft = compress_2D_odd(xfft, half_fft_W)
            yfft = compress_2D_odd(yfft, half_fft_W)

        # Compression.
        index_back_H_fft, index_back_W_fft = None, None
        if preserve_energy is not None and preserve_energy < 100.0:
            xfft, yfft = preserve_energy2D_symmetry(
                xfft, yfft, preserve_energy_rate=preserve_energy,
                is_debug=is_debug)
            # print("preserve energy timing: ", time.time() - start)
        elif index_back_W is not None and index_back_W > 0:
            is_fine_grained_sparsification = False  # this is for tests
            if is_fine_grained_sparsification:
                xfft_spectrum = get_spectrum(xfft)
                yfft_spectrum = get_spectrum(yfft)
                for _ in range(index_back_W):
                    xfft, xfft_spectrum = zero_out_min(xfft, xfft_spectrum)
                    yfft, yfft_spectrum = zero_out_min(yfft, yfft_spectrum)
            else:
                # At least one coefficient is removed.
                index_back_W_fft = int(
                    init_half_fft_W * (index_back_W / 100)) + 1
                xfft = compress_2D_odd_index_back(xfft, index_back_W_fft)
                yfft = compress_2D_odd_index_back(yfft, index_back_W_fft)

        out = torch.zeros([N, F, out_H, out_W], dtype=dtype, device=device)

        is_serial = False  # Serially convolve each input map with all filters.
        if is_serial:
            for nn in range(N):  # For each time-series in the batch.
                # Take one time series and unsqueeze it for broadcasting with
                # many filters.
                xfft_nn = xfft[nn].unsqueeze(0)
                out[nn] = correlate_fft_signals2D(
                    xfft=xfft_nn, yfft=yfft,
                    input_height=init_fft_H, input_width=init_fft_W,
                    init_fft_height=init_fft_H,
                    init_half_fft_width=init_half_fft_W,
                    out_height=out_H, out_width=out_W, is_forward=True)
                if bias is not None:
                    # Add the bias term for each filter (it has to be unsqueezed to
                    # the dimension of the out to properly sum up the values).
                    out[nn] += bias.unsqueeze(-1).unsqueeze(-1)
        else:
            # Convolve some part of the input batch with all filters.
            start = 0
            step = get_step_estimate(xfft, yfft, args)
            if bias is not None:
                unsqueezed_bias = bias.unsqueeze(-1).unsqueeze(-1)
            # For each slice of time-series in the batch.
            for start in range(start, N, step):
                stop = min(start + step, N)
                # Take one time series and unsqueeze it for broadcasting with
                # many filters.
                xfft_nn = xfft[start:stop].unsqueeze(dim=1)
                out[start:stop] = correlate_fft_signals2D(
                    xfft=xfft_nn, yfft=yfft,
                    input_height=init_fft_H, input_width=init_fft_W,
                    init_fft_height=init_fft_H,
                    init_half_fft_width=init_half_fft_W,
                    out_height=out_H, out_width=out_W, is_forward=False).sum(
                    dim=-3)
                if bias is not None:
                    # Add the bias term for each filter (it has to be unsqueezed to
                    # the dimension of the out to properly sum up the values).
                    out[start:stop] += unsqueezed_bias

        if (stride_H != 1 or stride_W != 1) and (
                stride_type is StrideType.STANDARD):
            out = out[:, :, ::stride_H, ::stride_W]

        print("out_size: ", out.size())
        print("stride: ", stride)
        print("W: ", W)
        print("WW: ", WW)

        if ctx:
            ctx.save_for_backward(xfft, yfft, to_tensor(H), to_tensor(HH),
                                  to_tensor(W), to_tensor(WW),
                                  to_tensor(init_fft_H), to_tensor(init_fft_W),
                                  is_manual,
                                  to_tensor(conv_index),
                                  to_tensor(compress_type.value),
                                  to_tensor(is_debug),
                                  to_tensor(preserve_energy),
                                  to_tensor(index_back_H_fft),
                                  to_tensor(index_back_W_fft),
                                  to_tensor(stride_H),
                                  to_tensor(stride_type.value)
                                  )

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

        xfft, yfft, H, HH, W, WW, init_fft_H, init_fft_W, is_manual, conv_index, compress_type, is_debug, preserve_energy, index_back_H_fft, index_back_W_fft, stride, stride_type = ctx.saved_tensors

        is_debug = True if is_debug == 1 else False
        if is_debug:
            print("execute backward pass")

        need_input_grad = ctx.needs_input_grad[0]
        need_filter_grad = ctx.needs_input_grad[1]
        need_bias_grad = ctx.needs_input_grad[2]

        for tensor_obj in ctx.saved_tensors:
            del tensor_obj
        omit_objs = [id(ctx)]

        del ctx

        dtype = xfft.dtype
        device = xfft.device

        is_manual[0] = 1  # Mark the manual execution of the backward pass.

        H = from_tensor(H)
        HH = from_tensor(HH)
        W = from_tensor(W)
        WW = from_tensor(WW)
        init_fft_H = from_tensor(init_fft_H)
        init_fft_W = from_tensor(init_fft_W)
        # is_manual is already a tensor.
        conv_index = from_tensor(conv_index)  # for the debug/test purposes
        compress_type = CompressType(from_tensor(compress_type))
        preserve_energy = from_tensor(preserve_energy)
        index_back_H_fft = from_tensor(index_back_H_fft)
        index_back_W_fft = from_tensor(index_back_W_fft)
        stride = from_tensor(stride)
        stride_type = StrideType(from_tensor(stride_type))

        if (stride is not None and stride > 1) and (
                stride_type is StrideType.STANDARD):
            N, F, HHH, WWW = dout.size()
            assert HHH == WWW
            out_H = out_W = W - WW + 1
            grad = torch.zeros(N, F, out_H, out_W)
            print("size grad: ", grad.size())
            print("size dout: ", dout.size())
            print("stride: ", stride)
            print("W: ", W)
            print("WW: ", WW)
            if out_H > HHH and out_W > WWW:
                grad[..., ::stride, ::stride] = dout
            else:
                assert out_H == HHH and out_W == WWW
            dout = grad
            del grad

        # The last dimension for xfft and yfft is the 2 element complex number.
        N, C, half_fft_compressed_H, half_fft_compressed_W, _ = xfft.shape
        F, C, half_fft_compressed_H, half_fft_compressed_W, _ = yfft.shape
        N, F, H_out, W_out = dout.shape
        if H_out != W_out:
            raise Exception("We only support square outputs.")

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

        if need_bias_grad:
            # The number of bias elements is equal to the number of filters.
            db = torch.zeros(F, dtype=dtype, device=device)

            # Calculate dB (the gradient for the bias term).
            # We sum up all the incoming gradients for each filter
            # bias (as in the affine layer).
            for ff in range(F):
                db[ff] += torch.sum(dout[:, ff, :])
        del dout

        doutfft = torch.rfft(padded_dout,
                             signal_ndim=Conv2dfftFunction.signal_ndim,
                             onesided=True)
        del padded_dout

        # the last dimension is for real and imaginary parts of the complex
        # numbers
        N, F, init_half_fft_H, init_half_fft_W, _ = doutfft.shape

        if half_fft_compressed_W < init_half_fft_W:
            doutfft = compress_2D_odd(doutfft, half_fft_compressed_W)

        if need_input_grad:
            # Initialize gradient output tensors.
            # the x used for convolution was with padding
            dx = torch.zeros([N, C, H, W], dtype=dtype, device=device)
            conjugate_yfft = pytorch_conjugate(yfft)
            del yfft
            is_serial = False  # Serially convolve each input map with all filters.
            if is_serial:
                for nn in range(N):
                    # Take one time series and unsqueeze it for broadcast with
                    # many gradients dout.
                    doutfft_nn = doutfft[nn].unsqueeze(1)
                    out = correlate_fft_signals2D(
                        xfft=doutfft_nn, yfft=conjugate_yfft,
                        input_height=init_fft_H, input_width=init_fft_W,
                        init_fft_height=init_half_fft_H,
                        init_half_fft_width=init_half_fft_W,
                        out_height=H, out_width=W,
                        is_forward=False)
                    # Sum over all the Filters (F).
                    out = torch.sum(out, dim=0)
                    out = torch.unsqueeze(input=out, dim=0)
                    dx[nn] = out
            else:
                # Convolve some part of the dout batch with all filters.
                start = 0
                step = 16
                # For each slice of time-series in the batch.
                for start in range(start, N, step):
                    stop = min(start + step, N)
                    doutfft_nn = doutfft[start:stop].unsqueeze(dim=2)
                    dx[start:stop] = correlate_fft_signals2D(
                        xfft=doutfft_nn, yfft=conjugate_yfft,
                        input_height=init_fft_H, input_width=init_fft_W,
                        init_fft_height=init_half_fft_H,
                        init_half_fft_width=init_half_fft_W,
                        out_height=H, out_width=W,
                        is_forward=False).sum(dim=1)
            del conjugate_yfft

        if need_filter_grad:
            dw = torch.zeros([F, C, HH, WW], dtype=dtype, device=device)
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
            # Serially convolve each input dout with all input maps xfft.
            is_serial = False
            if is_serial:
                for ff in range(F):
                    # doutfft_ff has as many channels as number of input filters.
                    doutfft_ff = doutfft[:, ff, ...].unsqueeze(1)
                    out = correlate_fft_signals2D(
                        xfft=xfft, yfft=doutfft_ff,
                        input_height=init_fft_H, input_width=init_fft_W,
                        init_fft_height=init_half_fft_H,
                        init_half_fft_width=init_half_fft_W,
                        out_height=HH, out_width=WW,
                        is_forward=False)
                    # For a given filter, we have to sum up all its contributions
                    # to all the input maps.
                    out = torch.sum(input=out, dim=0)
                    # `unsqueeze` the dimension 0 for the input data points (N).
                    out = torch.unsqueeze(input=out, dim=0)
                    # print("conv name: {}, out shape: {}, dw shape: {}, N: {}, C: {}, F: {}".format(
                    #     "conv"+str(conv_index), out.size(), dw.size(), str(N),
                    #     str(C), str(F)))
                    dw[ff] = out
            else:
                # Convolve some part of the dout batch with all input maps.
                start = 0
                step = 16
                for start in range(start, F, step):
                    stop = min(start + step, F)
                    # doutfft_ff has as many channels as number of input filters.
                    doutfft_ff = doutfft[:, start:stop, ...]
                    doutfft_ff = doutfft_ff.permute(1, 0, 2, 3, 4).unsqueeze(
                        dim=2)
                    dw[start:stop] = correlate_fft_signals2D(
                        xfft=xfft, yfft=doutfft_ff,
                        input_height=init_fft_H, input_width=init_fft_W,
                        init_fft_height=init_half_fft_H,
                        init_half_fft_width=init_half_fft_W,
                        out_height=HH, out_width=WW,
                        is_forward=False).sum(dim=1)
        del doutfft
        del xfft

        return dx, dw, db, None, None, None, None, None, None, None, None, \
               None, None


class Conv2dfft(Module):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=None, groups=None, bias=True,
                 filter_value=None, bias_value=None, is_manual=tensor([0]),
                 conv_index=None, args=None, out_size=None):
        """

        2D convolution using FFT implemented fully in PyTorch.

        :param in_channels: (int) – Number of channels in the input series.
        :param out_channels: (int) – Number of channels produced by the
        convolution (equal to the number of filters in the given conv layer).
        :param kernel_size: (int) - Size of the convolving kernel (the width and
        height of the filter).
        :param stride: what is the stride for the convolution (the pattern for
        omitted values).
        :param padding: the padding added to the (top and bottom) and to the
        (left and right) of the input signal.
        :param dilation: (int) – Spacing between kernel elements. Default: 1
        :param groups: (int) – Number of blocked connections from input channels
        to output channels. Default: 1
        :param bias: (bool) - add bias or not
        :param index_back: how many frequency coefficients should be
        discarded
        :param preserve_energy: how much energy should be preserved in the input
        image.
        :param out_size: what is the expected output size of the
        operation (when compression is used and the out_size is
        smaller than the size of the input to the convolution, then
        the max pooling can be omitted and the compression
        in this layer can serve as the frequency-based (spectral)
        pooling.
        :param filter_value: you can provide the initial filter, i.e.
        filter weights of shape (F, C, HH, WW), where
        F - number of filters, C - number of channels, HH - height of the
        filter, WW - width of the filter
        :param bias_value: you can provide the initial value of the bias,
        of shape (F,)
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param is_manual: to check if the backward computation of convolution
        was computed manually.
        :param conv_index: the index (number) of the convolutional layer.

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, we can generate the full output, and then remove the
        redundant elements according to the stride parameter. The more relevant
        method is to apply spectral pooling as a means to achieving the strided
        convolution.
        """
        super(Conv2dfft, self).__init__()

        self.args = args

        if dilation is not None and dilation > 1:
            raise NotImplementedError("dilation > 1 is not supported.")
        if groups is not None and groups > 1:
            raise NotImplementedError("groups > 1 is not supported.")

        self.is_filter_value = None  # Was the filter value provided?
        if filter_value is None:
            self.is_filter_value = False
            if out_channels is None or in_channels is None or \
                    kernel_size is None:
                raise ValueError("Either specify filter_value or provide all"
                                 "the required parameters (out_channels, "
                                 "in_channels and kernel_size) to generate the "
                                 "filter.")
            self.kernel_height, self.kernel_width = get_pair(kernel_size)
            self.filter = Parameter(
                torch.randn(out_channels, in_channels, self.kernel_height,
                            self.kernel_width, dtype=args.dtype))
        else:
            self.is_filter_value = True
            self.filter = filter_value
            out_channels = filter_value.shape[0]
            in_channels = filter_value.shape[1]
            self.kernel_height = filter_value.shape[2]
            self.kernel_width = filter_value.shape[3]

        # alias for the filter
        self.weight = self.filter
        self.is_bias_value = None  # Was the bias value provided.
        if bias_value is None:
            self.is_bias_value = False
            if bias is True:
                self.bias = Parameter(
                    torch.randn(out_channels, dtype=args.dtype))
            else:
                self.register_parameter('bias', None)
                self.bias = None
        else:
            self.is_bias_value = True
            self.bias = bias_value

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.stride = stride
        self.is_manual = is_manual
        self.conv_index = conv_index
        self.out_size = out_size

        if args is None:
            self.index_back = None
            self.preserve_energy = None
            self.is_debug = False
            self.next_power2 = False
            self.is_debug = False
            self.compress_type = CompressType.STANDARD
        else:
            self.index_back = args.index_back
            self.preserve_energy = args.preserve_energy
            self.next_power2 = args.next_power2
            self.is_debug = args.is_debug
            self.compress_type = args.compress_type

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_height * self.kernel_width
        stdv = 1. / math.sqrt(n)
        if self.is_filter_value is not None and self.is_filter_value is False:
            self.filter.data.uniform_(-stdv, stdv)
        if self.bias is not None and self.is_bias_value is False:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 2D convolution
        """
        # ctx, input, filter, bias, padding = (0, 0), stride = (1, 1),
        # args = None, out_size = None, is_manual = tensor([0]),
        # conv_index = None
        return Conv2dfftFunction.apply(
            input, self.filter, self.bias, self.padding, self.stride,
            self.args, self.out_size, self.is_manual, self.conv_index)


class Conv2dfftAutograd(Conv2dfft):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=None, groups=None, bias=True,
                 filter_value=None, bias_value=None, is_manual=tensor([0]),
                 conv_index=None, args=None, out_size=None):
        """
        2D convolution using FFT with backward pass executed via PyTorch's
        autograd.
        """
        super(Conv2dfftAutograd, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
            filter_value=filter_value, bias_value=bias_value,
            conv_index=conv_index, is_manual=is_manual, args=args,
            out_size=out_size)

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
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b,
        ... padding=(1, 1), stride=(2, 2))
        >>> result = conv.forward(input=x)
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
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b)
        >>> result = conv.forward(input=x)
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
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b, args=Arguments(index_back=1, preserve_energy=100))
        >>> result = conv.forward(input=x)
        >>> # expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        >>> expect = np.array([[[[21.75, 21.75], [18.75, 13.75]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test 2D convolution.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([0.0])
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b)
        >>> result = conv.forward(input=x)
        >>> expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test bias.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([-1.0])
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b)
        >>> result = conv.forward(input=x)
        >>> expect = np.array([[[[21.0, 21.0], [17., 13.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Don't use next power of 2.
        >>> # A single input map.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        >>> # A single filter.
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        >>> b = tensor([0.0])
        >>> conv = Conv2dfftAutograd(filter_value=y, bias=b)
        >>> result = conv.forward(input=x)
        >>> expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result,
        ... err_msg="The expected array x and computed y are not almost equal.")

        >>> # Test 2 channels and 2 filters.
        >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
        ... [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]])
        >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0],[3.0, -2.0]]],
        ... [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]])
        >>> b = tensor([0.0, 0.0])
        >>> conv = Conv2dfftAutograd(filter_value=y, bias_value=b)
        >>> result = conv.forward(input=x)
        >>> expect = np.array([[[[23.0, 32.0], [30., 4.]],[[11.0, 12.0],
        ... [13.0, -11.0]]]])
        >>> np.testing.assert_array_almost_equal(x=expect, y=result, decimal=5,
        ... err_msg="The expected array x and computed y are not almost equal.")
        """
        return Conv2dfftFunction.forward(
            ctx=None, input=input, filter=self.filter, bias=self.bias,
            padding=self.padding, stride=self.stride, is_manual=self.is_manual,
            conv_index=self.conv_index, args=self.args, out_size=self.out_size)


def test_run():
    torch.manual_seed(231)
    filter = np.array([[[[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [0.0, 1.0, -1.0]]]],
                      dtype=np.float32)
    filter = torch.from_numpy(filter)
    module = Conv2dfft(filter)
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
