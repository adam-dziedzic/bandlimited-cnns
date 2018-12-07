"""
Custom FFT based convolution that can rely on the autograd
(a tape-based automatic differentiation library that supports
all differentiable Tensor operations in pytorch).
"""
import math
import sys

import logging
import numpy as np
import torch
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter
# from memory_profiler import profile
from cnns.nnlib.pytorch_layers.gpu_profile import gpu_profile
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['GPU_DEBUG'] = '0'

from cnns.nnlib.pytorch_layers.pytorch_utils import complex_pad_simple
from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_fft_signals
from cnns.nnlib.pytorch_layers.pytorch_utils import fast_jmul
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy_simple
from cnns.nnlib.pytorch_layers.pytorch_utils import next_power2
from cnns.nnlib.pytorch_layers.pytorch_utils import preserve_energy_index_back
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate as conj
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import retain_big_coef_bulk
from cnns.nnlib.pytorch_layers.pytorch_utils import retain_low_coef
from cnns.nnlib.pytorch_layers.pytorch_utils import get_elem_size
from cnns.nnlib.pytorch_layers.pytorch_utils import get_tensors
from cnns.nnlib.pytorch_layers.pytorch_utils import get_step_estimate
from cnns.nnlib.pytorch_layers.pytorch_utils import cuda_mem_empty
from cnns.nnlib.pytorch_layers.pytorch_utils import cuda_mem_show
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
# from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul_cpp
from cnns.nnlib.utils.general_utils import additional_log_file
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import plot_signal_freq
from cnns.nnlib.utils.general_utils import plot_signal_time
from cnns.nnlib.utils.arguments import Arguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class Conv1dfftFunction(torch.autograd.Function):
    """
    Implement the 1D convolution via FFT with compression of the input map and
    the filter.
    """
    signal_ndim = 1

    @staticmethod
    # @profile
    def forward(ctx, input, filter, bias=None, padding=0, stride=1,
                args=Arguments(), out_size=None, is_manual=tensor([0]),
                conv_index=None):
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
        :param stride: what is the stride for the convolution (the pattern for
        omitted values).
        :param index_back: how many last elements in the fft-ed signal to
        discard.
        :param preserve_energy: how much energy of the input should be
        preserved.
        :param out_size: what is the expected output size - one can disard the
        elements in the frequency domain and do the spectral pooling within the
        convolution.
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
        # if is_debug:
        #     gpu_profile(frame=sys._getframe(), event='line', arg=None)

        # reverse which side of a signal is cut off: head or tail
        is_lead_reversed = False

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
            print(f"execute forward pass 1D for layer index {conv_index}")
            data_point = 0
            data_channel = 0
            x_signal = input[data_point, data_channel]
            np_signal = x_signal.cpu().detach().numpy()
            print("np_signal: ", np_signal)
            plot_signal_time(np_signal,
                             title=f"data_point {data_point}, "
                             f"data_channel {data_channel},"
                             f" conv {conv_index} time domain",
                             xlabel="Time")
            filter_bank = 0
            filter_channel = 0
            filter_signal = filter[filter_bank, filter_channel]
            np_filter = filter_signal.cpu().detach().numpy()
            print("np_filter: ", np_filter)
            plot_signal_time(np_filter,
                             title=f"filter bank {filter_bank}, "
                             f"filter channel {filter_channel},"
                             f" conv {conv_index} time domain",
                             xlabel="Time")
            cuda_mem_show(info="forward start")

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
            out_W += 2 * padding_count  # padding is applied to both sides

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
        We can add the [WW-1] at the other side of [W]:
        [WW-1][W][WW-1]
        and also have to account for the convolution with flowing back gradient:
        [WW-1][W][WW-1][WWW-1]
        
        Final padding:
        W: [WW-1][W][WW-1][WWW-1]
        WW: [WW-1][WW][WW-1][W+WW-1-WW]
        WWW: [WW-1][WWW][WW-1][W-1]
        """
        filter_pad = WW - 1  # padding from the filter
        # padding from the flowing back gradient fft-ed map
        dout_pad = out_W - 1
        init_fft_size = W + 2 * padding_count + 2 * filter_pad + dout_pad
        if use_next_power2:
            fft_size = next_power2(init_fft_size)
        else:
            fft_size = init_fft_size

        # How many padded (zero) values there are because of going to the next
        # power of 2?
        fft_padding_x = fft_size - init_fft_size

        # Pad only the dimensions for the time-series - the width dimension
        # (and neither data points nor the channels). We pad the input signal
        # on both sides with (filter_size - 1) to cater for the backward pass
        # where the required form of the dout is [WW-1][WWW][WW-1].

        left_x_pad = filter_pad + padding_count
        right_x_pad = padding_count + filter_pad + dout_pad + fft_padding_x
        input = torch_pad(input, (left_x_pad, right_x_pad), 'constant', 0)
        if is_debug:
            cuda_mem_show(info="input pad")

        # fft of the input signals.
        xfft = torch.rfft(input, signal_ndim=Conv1dfftFunction.signal_ndim,
                          onesided=True)
        del input

        if is_debug:
            data_point = 0
            data_channel = 0
            xfft_signal = xfft[data_point, data_channel]
            spectrum = get_spectrum(xfft_signal)
            data_spectrum_np = spectrum.cpu().numpy()
            print("data spectrum np: ", data_spectrum_np)
            plot_signal_freq(data_spectrum_np,
                             title=f"data_point {data_point}, "
                             f"data_channel {data_channel},"
                             f" conv {conv_index} spectral",
                             xlabel="Frequency")
            cuda_mem_show(info="input fft")

        # The last dimension (-1) has size 2 as it represents the complex
        # numbers with real and imaginary parts. The last but one dimension
        # (-2) represents the length of the signal in the frequency domain.
        init_half_fft_size = xfft.shape[-2]
        half_fft_compressed_size = None
        index_back_fft = None
        if index_back is not None and index_back > 0:
            # At least one coefficient is removed.
            index_back_fft = int(init_half_fft_size * (index_back / 100)) + 1

        if compress_type is CompressType.STANDARD:
            if preserve_energy is not None and preserve_energy < 100:
                index_back_fft = preserve_energy_index_back(xfft,
                                                            preserve_energy)
            if out_size is not None:
                # We take onesided fft so the output after inverse fft should
                # be out size, thus the representation in spectral domain is
                # twice smaller than the one in time domain. We require at least
                # one fft coefficient retained.
                half_fft_compressed_size = out_size // 2 + 1

        # if is_debug:
        #     cuda_mem_show(info="compute index back")

        if index_back_fft is not None:
            # At least one frequency coefficient has to be removed.
            index_back_fft = min(init_half_fft_size - 1, index_back_fft)
            if compress_type is CompressType.STANDARD:
                half_fft_compressed_size = init_half_fft_size - index_back_fft

        if is_debug is True:
            full_energy, squared = get_full_energy_simple(xfft)
            cuda_mem_show(info="get full energy")

            del squared

            cuda_mem_show(info="delete amplitudes")

        if half_fft_compressed_size is not None:
            # We request at least one coefficient to remain.
            half_fft_compressed_size = max(1, half_fft_compressed_size)
            # xfft = xfft[:, :, :half_fft_compressed_size, :]
            if is_lead_reversed:
                # The last dimension if for the real and imaginary part of a
                # complex number.
                xfft_compressed = xfft[..., -half_fft_compressed_size:, :]
            else:
                xfft_compressed = xfft.narrow(dim=2, start=0,
                                              length=half_fft_compressed_size)
            del xfft
            xfft = xfft_compressed

        if is_debug:
            cuda_mem_show(info="compress input")

        # fft of the filters.
        fft_size_filter = fft_size
        if compress_type is CompressType.NO_FILTER and (
                half_fft_compressed_size is not None):
            # At least 1 coefficient in the filter.
            fft_size_filter = max(1, (half_fft_compressed_size - 1) * 2)

        fft_padding_filter = fft_size_filter - (WW + 2 * filter_pad)
        # We have to pad the filter (at least the filter size - 1).
        # fft_padding_filter can be negative number if
        right_filter_pad = max(filter_pad, filter_pad + fft_padding_filter)
        filter = torch_pad(filter, (filter_pad, right_filter_pad),
                           'constant', 0)

        if is_debug:
            cuda_mem_show(info="filter pad")

        yfft = torch.rfft(filter, signal_ndim=Conv1dfftFunction.signal_ndim,
                          onesided=True)
        del filter
        if is_debug:
            filter_bank = 0
            filter_channel = 0
            yfft_signal = yfft[filter_bank, filter_channel]
            spectrum = get_spectrum(yfft_signal)
            filter_spectrum_np = spectrum.cpu().numpy()
            print("filter_spectrum_np: ", filter_spectrum_np)
            plot_signal_freq(filter_spectrum_np,
                             title=f"filter bank {filter_bank}, "
                             f"filter channel {filter_channel},"
                             f" conv {conv_index} spectral",
                             xlabel="Frequency")
            cuda_mem_show(info="filter fft")

        if is_debug:
            print("conv_name," + "conv" + str(conv_index))

        if half_fft_compressed_size is not None:
            # yfft = yfft[:, :, :half_fft_compressed_size, :]
            if is_lead_reversed:
                yfft_compressed = yfft[..., -half_fft_compressed_size:, :]
            else:
                yfft_compressed = yfft.narrow(dim=-2, start=0,
                                              length=half_fft_compressed_size)
            del yfft
            yfft = yfft_compressed

        if is_debug:
            cuda_mem_show(info="compress filter")

        if compress_type is CompressType.BIG_COEFF:
            if preserve_energy is not None and preserve_energy < 100:
                xfft = retain_big_coef_bulk(xfft,
                                            preserve_energy=preserve_energy)
                yfft = retain_big_coef_bulk(yfft,
                                            preserve_energy=preserve_energy)
            elif index_back_fft is not None and index_back_fft > 0:
                xfft = retain_big_coef_bulk(xfft, index_back=index_back_fft)
                yfft = retain_big_coef_bulk(yfft, index_back=index_back_fft)
        elif compress_type is CompressType.LOW_COEFF:
            if preserve_energy is not None and preserve_energy < 100:
                xfft = retain_low_coef(xfft, preserve_energy=preserve_energy)
                yfft = retain_low_coef(yfft, preserve_energy=preserve_energy)
            elif index_back_fft is not None and index_back_fft > 0:
                xfft = retain_low_coef(xfft, index_back=index_back_fft)
                yfft = retain_low_coef(yfft, index_back=index_back_fft)

        if is_debug is True:
            if half_fft_compressed_size is None:
                percent_retained_signal = 100
            else:
                percent_retained_signal = 100 * (
                        half_fft_compressed_size / init_half_fft_size)
            preserved_energy, squared = get_full_energy_simple(xfft)
            del squared
            # The xfft can be only with zeros thus the full energy is zero.
            percent_preserved_energy = 0.0

            if full_energy > 0.0:
                percent_preserved_energy = preserved_energy / full_energy * 100
            msg = "conv_name," + "conv" + str(
                conv_index) + ",index_back_fft," + str(
                index_back_fft) + ",raw signal length," + str(
                W) + ",fft_size," + str(
                fft_size) + ",init_half_fft_size," + str(
                init_half_fft_size) + ",half_fft_compressed_size," + str(
                half_fft_compressed_size) + (
                      ",percent of preserved energy,") + str(
                percent_preserved_energy) + (
                      ",percent of retained signal,") + str(
                percent_retained_signal) + ",fft_size_filter," + str(
                fft_size_filter) + ",half_filter_fft_size," + str(
                yfft.shape[-2]) + ",preserve_energy," + str(
                preserve_energy) + ",new-line"
            print(msg)
            with open(additional_log_file, "a") as file:
                file.write(msg + "\n")

        # # 2 is for the complex numbers
        # output = torch.zeros([N, F, xfft.size()[-2], 2], dtype=xfft.dtype,
        #                      device=xfft.device)
        #
        # for nn in range(N):  # For each time-series in the batch.
        #     # Take one time series and un-squeeze it for broadcasting with
        #     # many filters.
        #     xfft_nn = xfft[nn].unsqueeze(0)
        #     out = complex_mul_cpp(xfft_nn, pytorch_conjugate(yfft))
        #     """
        #     Sum up the elements from computed output maps for each input
        #     channel. Each output map has as many channels as the number of
        #     filters. Each filter contributes one channel for the output map.
        #     """
        #     # Sum across the input channels.
        #     out = torch.sum(input=out, dim=-3)
        #     output[nn] = out.unsqueeze(0)
        # output = complex_pad_simple(xfft=output, fft_size=fft_size)
        # output = torch.irfft(
        #     input=output, signal_ndim=signal_ndim, signal_sizes=(fft_size,))
        # if output.shape[-1] > out_W:
        #     output = output[..., :out_W]
        # elif output.shape[-1] < out_W:
        #     output = torch_pad(output, (0, out_W - output.shape[-1]))
        #
        # if bias is not None:
        #     # Add the bias term for each filter.
        #     # Bias has to be unsqueezed to the dimension of the out to
        #     # properly sum up the values.
        #     output += bias.unsqueeze(1)

        output = torch.zeros([N, F, out_W], dtype=dtype, device=device)

        if args.conv_exec_type is ConvExecType.SERIAL:
            # Serially convolve each input map with all filters.
            for nn in range(N):  # For each time-series in the batch.
                # Take one time series and un-squeeze it for broadcasting with
                # many filters.
                xfft_nn = xfft[nn].unsqueeze(0)
                out_fft = correlate_fft_signals(
                    xfft=xfft_nn, yfft=yfft, fft_size=fft_size)
                if out_fft.shape[-1] > out_W:
                    # out_fft = out_fft[..., :out_W]
                    out_fft = out_fft.narrow(dim=-1, start=0, length=out_W)
                elif out_fft.shape[-1] < out_W:
                    out_fft = torch_pad(out_fft,
                                        (0, out_W - out_fft.shape[-1]))

                """
                Sum up the elements from computed output maps for each input 
                channel. Each output map has as many channels as the number of 
                filters. Each filter contributes one channel for the output map. 
                """
                # Sum the input channels.
                out_fft = torch.sum(input=out_fft, dim=1)
                # `unsqueeze` the dimension for channels.
                out_fft = torch.unsqueeze(input=out_fft, dim=0)
                output[nn] = out_fft

                if bias is not None:
                    # Add the bias term for each filter.
                    # Bias has to be unsqueezed to the dimension of the
                    # out to properly sum up the values.
                    output[nn] += bias.unsqueeze(1)
        else:
            # Convolve some part of the input batch with all filters.
            start = 0
            # step = get_step_estimate(xfft, yfft, args.memory_size)
            step = int(args.min_batch_size)
            if bias is not None:
                unsqueezed_bias = bias.unsqueeze(-1)
            # For each slice of time-series in the batch.
            for start in range(start, N, step):
                stop = min(start + step, N)
                # Take one time series and unsqueeze it for broadcasting with
                # many filters.
                xfft_nn = xfft[start:stop].unsqueeze(dim=1)
                out = correlate_fft_signals(
                    xfft=xfft_nn, yfft=yfft, fft_size=fft_size)
                del xfft_nn
                if out.shape[-1] > out_W:
                    # out_fft = out_fft[..., :out_W]
                    out = out.narrow(dim=-1, start=0, length=out_W)
                elif out.shape[-1] < out_W:
                    out = torch_pad(out, (0, out_W - out.shape[-1]))
                out = out.sum(dim=-2)  # sum over channels C
                output[start:stop] = out
                del out
                if bias is not None:
                    # Add the bias term for each filter (it has to be unsqueezed
                    # to the dimension of the output to properly sum up the
                    # values).
                    output[start:stop] += unsqueezed_bias

        if is_debug:
            cuda_mem_show(info="compute output")

        # TODO: how to compute the backward pass for the strided FFT
        # convolution?
        # Add additional zeros in the places of the output that were removed
        # by striding.
        if stride is not None and stride > 1 and (
                stride_type is StrideType.STANDARD):
            output = output[:, :, 0::stride]

        if ctx:
            ctx.run_args = args
            ctx.save_for_backward(xfft, yfft, to_tensor(W), to_tensor(WW),
                                  to_tensor(fft_size), is_manual,
                                  to_tensor(conv_index),
                                  to_tensor(compress_type.value),
                                  to_tensor(is_debug),
                                  to_tensor(preserve_energy),
                                  to_tensor(index_back_fft),
                                  to_tensor(args.memory_size))

        # print("context type: ", type(ctx))
        # tensors = get_tensors(only_cuda=False, is_debug=True)
        # print("tensors: ", ",".join([str(tensor) for tensor in tensors]))

        if is_debug:
            cuda_mem_show(info="forward end")

        return output

    @staticmethod
    # @profile
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
        # print("manual backward pass")
        xfft, yfft, W, WW, fft_size, is_manual, conv_index, compress_type, is_debug, preserve_energy, index_back_fft, memory_size = ctx.saved_tensors
        need_input_grad = ctx.needs_input_grad[0]
        need_filter_grad = ctx.needs_input_grad[1]
        need_bias_grad = ctx.needs_input_grad[2]
        for tensor_obj in ctx.saved_tensors:
            del tensor_obj
        omit_objs = [id(ctx)]
        args = ctx.run_args
        conv_index = from_tensor(conv_index)  # for the debug/test purposes

        is_debug = from_tensor(is_debug)
        is_debug = True if is_debug == 1 else False
        if is_debug:
            print("Conv layer index: ", conv_index)
            dout_point = 0
            dout_channel = 0
            dout_signal = dout[dout_point, dout_channel]
            np_dout = dout_signal.cpu().detach().numpy()
            print("np_dout: ", np_dout)
            plot_signal_time(np_dout,
                             title=f"dout_point {dout_point}, "
                             f"dout_channel {dout_channel},"
                             f" conv {conv_index} time domain",
                             xlabel="Time")
            total_size = 0
            for tensor_obj in ctx.saved_tensors:
                total_size += tensor_obj.numel() * get_elem_size(tensor_obj)
            print("size of the context: ", total_size)

        del ctx

        dtype = xfft.dtype
        device = xfft.device

        is_manual[0] = 1  # Mark the manual execution of the backward pass.
        W = from_tensor(W)
        WW = from_tensor(WW)
        fft_size = from_tensor(fft_size)
        # is_manual is already a tensor.
        compress_type = CompressType(from_tensor(compress_type))
        preserve_energy = from_tensor(preserve_energy)
        index_back_fft = from_tensor(index_back_fft)
        memory_size = from_tensor(memory_size)

        if is_debug:
            print("execute backward pass 1D")
            cuda_mem_show(info="backward start", omit_objs=omit_objs)

        # if is_debug:
        #     gpu_profile(frame=sys._getframe(), event='line', arg=None)

        # The last dimension (_) for xfft and yfft is the 2 element complex
        # number.
        N, C, half_fft_compressed_size, _ = xfft.shape
        F, C, half_fft_compressed_size, _ = yfft.shape
        N, F, out_W = dout.shape

        dx = dw = db = None

        # Gradient for the bias.
        if need_bias_grad is True:
            # The number of bias elements is equal to the number of filters.
            db = torch.zeros(F, device=device, dtype=dtype)

            # Calculate dB (the gradient for the bias term).
            # We sum up all the incoming gradients for each filter
            # bias (as in the affine layer).
            for ff in range(F):
                db[ff] += torch.sum(dout[:, ff, :])

        # fft of the gradient.
        fft_size_grad = fft_size
        if compress_type is CompressType.NO_FILTER and (
                half_fft_compressed_size is not None):
            # Decreases the required padding for grad thus we do not have to
            # compress the signal in frequency domain (by removing the
            # coefficients from the end of the signal).
            fft_size_grad = (half_fft_compressed_size - 1) * 2

        # Take the fft of dout (the gradient of the output of the forward pass).
        # We have to pad the flowing back gradient in the time (spatial) domain,
        # since it does not give correct results even for the case without
        # compression if we pad in the spectral (frequency) domain.
        # We pad both sides of the dout gradient. The left side is padded by
        # (WW-1), the right side is padded also by (WW-1) and the additional
        # zeros that are required to fill in the init_fft_size.
        filter_pad = WW - 1
        left_pad = filter_pad
        # out_W is the length of dout as well.
        fft_pad = fft_size_grad - (filter_pad + out_W + filter_pad)
        right_pad = filter_pad + fft_pad
        dout = torch_pad(dout, (left_pad, right_pad), 'constant', 0)

        if is_debug:
            cuda_mem_show(info="gradient pad", omit_objs=omit_objs)

        doutfft = torch.rfft(dout, signal_ndim=Conv1dfftFunction.signal_ndim,
                             onesided=True)
        del dout

        if is_debug:
            dout_point = 0
            dout_channel = 0
            dout_signal = doutfft[dout_point, dout_channel]
            dout_spectrum = get_spectrum(dout_signal)
            dout_spectrum_np = dout_spectrum.cpu().numpy()
            print("dout spectrum np: ", dout_spectrum_np)
            plot_signal_freq(dout_spectrum_np,
                             title=f"dout_point {dout_point}, "
                             f"dout_channel {dout_channel},"
                             f" conv {conv_index} spectral",
                             xlabel="Frequency")
            cuda_mem_show(info="gradient fft", omit_objs=omit_objs)

        # If the compression was done in the forward pass, then we have to
        # compress the pure fft-ed version of the flowing back gradient:
        # doutftt.
        init_half_fft_grad_size = doutfft.shape[-2]

        if half_fft_compressed_size < init_half_fft_grad_size:
            doutfft_compressed = doutfft.narrow(dim=-2, start=0,
                                                length=half_fft_compressed_size)
            del doutfft
            doutfft = doutfft_compressed
            # doutfft = doutfft[:, :, :half_fft_compressed_size, :]

        if is_debug:
            cuda_mem_show(info="compress gradient", omit_objs=omit_objs)

        elif compress_type is CompressType.BIG_COEFF:
            if preserve_energy is not None and preserve_energy < 100:
                doutfft = retain_big_coef_bulk(doutfft,
                                               preserve_energy=preserve_energy)
            elif index_back_fft is not None and index_back_fft > 0:
                doutfft = retain_big_coef_bulk(doutfft,
                                               index_back=index_back_fft)
        elif compress_type is CompressType.LOW_COEFF:
            if preserve_energy is not None and preserve_energy < 100:
                doutfft = retain_low_coef(doutfft,
                                          preserve_energy=preserve_energy)
            elif index_back_fft is not None and index_back_fft > 0:
                doutfft = retain_low_coef(doutfft, index_back=index_back_fft)

        if need_input_grad is True:
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
            dx = torch.zeros([N, C, W], dtype=dtype, device=device)
            conjugate_yfft = pytorch_conjugate(yfft)
            del yfft
            if args.conv_exec_type is ConvExecType.SERIAL:
                # Serially convolve each input map with all filters.
                for nn in range(N):
                    # Take one time series and unsqueeze it for broadcast with
                    # many gradients dout. We assign 1 to the input channel C.
                    # dout: (N, F, WWW)
                    # weight w: (F, C, WW)
                    # grad dx: (N, C, W)
                    # A single input map was convolved with many filters F.
                    # We choose a single output map dout[nn], and unsqueeze it for
                    # the input channel dimension 1. Then we sum up over all filter
                    # F, but also produce gradients for all the channels C.
                    doutfft_nn = doutfft[nn, :, :].unsqueeze(1)
                    out = correlate_fft_signals(
                        xfft=doutfft_nn, yfft=conjugate_yfft,
                        fft_size=fft_size)
                    start_index = 2 * filter_pad
                    # print("start index: ", start_index)
                    out = out[:, :, start_index: start_index + W]
                    # Sum over all the Filters (F).
                    out = torch.sum(out, dim=0)
                    out = torch.unsqueeze(input=out, dim=0)
                    dx[nn] = out
            else:
                # Convolve some part of the dout batch with all filters.
                start = 0
                # step = 16
                # step = get_step_estimate(doutfft, conjugate_yfft, memory_size)
                step = int(args.min_batch_size)
                doutfft = doutfft.unsqueeze(dim=2)
                # For each slice of time-series in the batch.
                for start in range(start, N, step):
                    stop = min(start + step, N)
                    doutfft_nn = doutfft[start:stop]
                    # print("doutfft_nn size: ", doutfft_nn.size())
                    # print("conjugateyfft size: ", conjugate_yfft.size())
                    out = correlate_fft_signals(
                        xfft=doutfft_nn, yfft=conjugate_yfft,
                        fft_size=fft_size)
                    start_index = 2 * filter_pad
                    # print("start index: ", start_index)
                    out = out[..., start_index: start_index + W]
                    # print("out size: ", out.size())
                    # Sum over all the Filters (F).
                    out = torch.sum(out, dim=1)
                    # print("out after sum size: ", out.size())
                    dx[start:stop] = out

            del conjugate_yfft

            if is_debug:
                cuda_mem_show(info="after gradient input",
                              omit_objs=omit_objs)

        if need_filter_grad is True:
            dw = torch.zeros([F, C, WW], dtype=dtype, device=device)
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
            if args.conv_exec_type is ConvExecType.SERIAL:
                for ff in range(F):
                    # Gather all the contributions to the output that were caused
                    # by a given filter.
                    doutfft_ff = doutfft[:, ff, :].unsqueeze(1)
                    out = correlate_fft_signals(
                        xfft=xfft, yfft=doutfft_ff, fft_size=fft_size,
                        signal_ndim=Conv1dfftFunction.signal_ndim)
                    out = out[:, :, :WW]
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
                # step = 16
                # step = get_step_estimate(xfft, doutfft, memory_size=memory_size)
                step = int(args.min_batch_size)
                # print("doutfft size: ", doutfft.size())
                if len(doutfft.shape) == 4:  # we did not need grad for input
                    doutfft = doutfft.unsqueeze(dim=2)
                doutfft = doutfft.permute(1, 0, 2, 3, 4)

                for start in range(start, F, step):
                    stop = min(start + step, F)
                    # doutfft_ff has as many channels as number of input filters.
                    doutfft_ff = doutfft[start:stop]
                    # print("doutfft_ff size: ", doutfft_ff.size())
                    # print("xfft size: ", xfft.size())
                    out = correlate_fft_signals(
                        xfft=xfft, yfft=doutfft_ff, fft_size=fft_size,
                        signal_ndim=Conv1dfftFunction.signal_ndim)
                    out = out[..., :WW]
                    # print("out size: ", out.size())
                    # For a given filter, we have to sum up all its contributions
                    # to all the input maps.
                    out = out.sum(dim=1)
                    dw[start:stop] = out
            del xfft

            if is_debug:
                cuda_mem_show(info="after gradient filter",
                              omit_objs=omit_objs)

        if is_debug:
            cuda_mem_show(info="backward end", omit_objs=omit_objs)

        return dx, dw, db, None, None, None, None, None, None


class Conv1dfft(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=None, groups=None, bias=True,
                 filter_value=None, bias_value=None, is_manual=tensor([0]),
                 conv_index=None, args=Arguments(), out_size=None):
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
        :param preserve_energy: how much energy should be preserved in the input
        signal.
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
        :param conv_index: the index (number) of the convolutional layer.
        :param is_complex_pad: is padding applied to the complex representation
        of the input signal and filter before the reverse fft is applied.
        :param is_debug: is this the debug mode execution?
        :param compress_type: the type of FFT compression, NO_FILTER - do not
        compress the filter. BIG_COEF: preserve only the largest coefficients
        in the frequency domain.
        :param dtype: the data type of PyTorch tensors.

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, we can generate the full output, and then remove the
        redundant elements according to the stride parameter. We have to figure
        out how to run the backward pass for this strided FFT-based convolution.
        """
        super(Conv1dfft, self).__init__()

        self.args = args

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
            self.filter = Parameter(
                torch.randn(out_channels, in_channels, kernel_size,
                            dtype=args.dtype))
        else:
            self.is_filter_value = True
            self.filter = filter_value
            out_channels = filter_value.shape[0]
            in_channels = filter_value.shape[1]
            kernel_size = filter_value.shape[2]

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
        self.out_size = out_size
        self.stride = stride
        self.is_manual = is_manual
        self.conv_index = conv_index

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        # We have only a single kernel size for 1D convolution.
        n *= self.kernel_size
        stdv = 1. / math.sqrt(n)
        if self.is_filter_value is False:
            self.filter.data.uniform_(-stdv, stdv)
        if self.bias is not None and self.is_bias_value is False:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return Conv1dfftFunction.apply(
            input, self.filter, self.bias, self.padding, self.stride,
            self.args, self.out_size, self.is_manual, self.conv_index)


class Conv1dfftAutograd(Conv1dfft):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 index_back=None, preserve_energy=100, out_size=None,
                 filter_value=None, bias_value=None, use_next_power2=False,
                 is_manual=tensor([0]), conv_index=None, is_complex_pad=True,
                 is_debug=False, compress_type=CompressType.STANDARD):
        super(Conv1dfftAutograd, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            index_back=index_back, out_size=out_size, filter_value=filter_value,
            bias_value=bias_value, use_next_power2=use_next_power2,
            conv_index=conv_index, preserve_energy=preserve_energy,
            is_debug=is_debug, compress_type=compress_type, is_manual=is_manual,
            dilation=dilation, groups=groups, is_complex_pad=is_complex_pad)

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
        >>> expected_result = [3.666667, 7.333333]
        >>> conv = Conv1dfftAutograd(filter_value=torch.from_numpy(y),
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
        >>> conv = Conv1dfftAutograd(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        return Conv1dfftFunction.forward(
            ctx=None, input=input, filter=self.filter, bias=self.bias,
            padding=self.padding, stride=self.stride,
            index_back=self.index_back, preserve_energy=self.preserve_energy,
            out_size=self.out_size, use_next_power2=self.use_next_power2,
            is_manual=self.is_manual, conv_index=self.conv_index,
            is_debug=self.is_debug, compress_type=self.compress_type)


def test_run():
    torch.manual_seed(231)
    filter = np.array([[[1., 2., 3.]]], dtype=np.float32)
    filter = torch.from_numpy(filter)
    module = Conv1dfft(filter_value=filter)
    print()
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)
    assert module.is_manual[0] == 1
    print("The manual backprop was executed.")


class Conv1dfftSimple(Conv1dfftAutograd):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=None, padding=None, bias=True, index_back=None,
                 out_size=None, filter_value=None, bias_value=None,
                 is_debug=False):
        super(Conv1dfftSimple, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
            index_back=index_back, out_size=out_size, filter_value=filter_value,
            bias_value=bias_value, is_debug=is_debug)

    def forward(self, input):
        """
        This is a simple manual implementation of the forward pass with the
        pytroch autograd used for the backward pass.

        :param input: the input map (e.g., a 1D signal such as time-series
        data.)
        :return: the result of 1D convolution.

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
        >>> conv = Conv1dfftSimple(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        input_size = input.shape[-1]
        fft_size = input_size + self.kernel_size - 1
        out_size = input_size - self.kernel_size + 1

        # Pad and transform the input.
        input = torch_pad(input, (0, self.kernel_size - 1))
        input = torch.rfft(input, 1)
        # Pad and transform the filters.
        filter = torch_pad(self.filter, (0, input_size - 1))
        filter = torch.rfft(filter, 1)

        if self.index_back is not None and self.index_back > 0:
            # 4 dims: batch, channel, time-series, complex values.
            input_size = input.shape[-2]
            index_back = int(input_size * (self.index_back / 100)) + 1
            input = input[..., :-index_back, :]
            filter = filter[..., :-index_back, :]

        input = input.unsqueeze(1)
        out = fast_jmul(input, conj(filter))
        if out.shape[-1] < fft_size:
            out = complex_pad_simple(out, fft_size)
        out = torch.irfft(out, 1, signal_sizes=(fft_size,))
        if out.shape[-1] > out_size:
            out = out[..., :out_size]

        """
        Sum up the elements from computed output maps for each input 
        channel. Each output map has as many channels as the number of 
        filters. Each filter contributes one channel for the output map. 
        """
        # Sum the input channels.
        out = torch.sum(input=out, dim=2)
        if self.bias is not None:
            # Add the bias term for each filter.
            # Bias has to be unsqueezed to the dimension of the out to
            # properly sum up the values.
            out += self.bias.unsqueeze(1)

        return out


class Conv1dfftSimpleForLoop(Conv1dfftAutograd):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=None, padding=None, bias=True, index_back=None,
                 preserve_energy=None, out_size=None, filter_value=None,
                 bias_value=None, conv_index=None, use_next_power2=False,
                 is_complex_pad=True):
        """
        A simple implementation of 1D convolution with a single for loop where
        we iterate over the input signals and for each of them convolve it with
        all the filters.

        :param index_back: this is changed. This is the percentage from 0 to 100
        of the size of the input signal. This percentage of the input signal is
        the size - number of frequencies that will be discarded in the frequency
        domain. Calculations:
        index_back = int(input_size * (self.index_back / 100) // 2) + 1
        """
        super(Conv1dfftSimpleForLoop, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias, index_back=index_back, preserve_energy=preserve_energy,
            out_size=out_size, filter_value=filter_value, bias_value=bias_value,
            conv_index=conv_index, use_next_power2=use_next_power2,
            is_complex_pad=is_complex_pad)

    def forward(self, input):
        """
        This is a simple manual implementation of the forward pass with the
        pytroch autograd used for the backward pass.
        :param input: the input map (e.g., a 1D signal such as time-series
        data.)
        :return: the result of 1D convolution.
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
        >>> conv = Conv1dfftSimpleForLoop(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))
        """
        batch_num = input.shape[0]
        filter_num = self.filter.shape[0]
        input_size = input.shape[-1]
        fft_size = input_size + self.kernel_size - 1
        if self.use_next_power2 is True:
            fft_size = next_power2(fft_size)
        init_fft_size = fft_size
        out_size = input_size - self.kernel_size + 1

        # Pad and transform the input.
        input = torch_pad(input, (0, fft_size - input_size))
        input = torch.rfft(input, 1)
        # Pad and transform the filters.
        filter = torch_pad(self.filter, (0, fft_size - self.kernel_size))
        filter = torch.rfft(filter, 1)

        # Change from the percentage of how many coefficient should be discarded
        # to the the actual number of coefficients to discard.
        if self.index_back is not None and self.index_back > 0:
            if self.preserve_energy is not None and self.preserve_energy < 100:
                raise AttributeError(
                    "Choose either preserve_energy or index_back or output size.")
            self.index_back = int(input.shape[-2] * (self.index_back / 100)) + 1

        if self.preserve_energy is not None and self.preserve_energy < 100:
            self.index_back = preserve_energy_index_back(input,
                                                         self.preserve_energy)

        if self.index_back is not None and self.index_back > 0:
            # At least 1 frequency coefficient has to remain.
            init_half_fft_size = input.shape[-2]
            self.index_back = min(init_half_fft_size - 1, self.index_back)
            full_energy, _ = get_full_energy_simple(x=input)
            input = input[..., :-self.index_back, :]
            preserved_energy, _ = get_full_energy_simple(x=input)
            filter = filter[..., :-self.index_back, :]
            # The initial input fft_size has to be at least 1.
            fft_size = max((input.shape[-2] - 1) * 2, 1)

            percent_retained_signal = (
                                              init_half_fft_size - self.index_back) / init_half_fft_size * 100
            percent_preserved_energy = preserved_energy / full_energy * 100
            msg = "conv_name," + "conv" + str(
                self.conv_index) + ",index_back," + str(
                self.index_back) + ",init_fft_size," + str(
                init_fft_size) + ",raw signal length," + str(
                input_size) + ",init_half_fft_size," + str(
                init_half_fft_size) + ",percent of preserved energy," + str(
                percent_preserved_energy) + (
                      ",percent of retained signal,") + str(
                percent_retained_signal)
            print(msg)
            with open(additional_log_file, "a") as file:
                file.write(msg + "\n")

        output = torch.zeros([batch_num, filter_num, out_size],
                             dtype=input.dtype, device=input.device)

        for batch_idx in range(batch_num):
            # Broadcast each 1D signal with all filters.
            signal = input[batch_idx].unsqueeze(0)
            if self.is_complex_pad is True:
                out = correlate_fft_signals(xfft=signal, yfft=filter,
                                            fft_size=fft_size)
            else:
                out = fast_jmul(signal, conj(filter))
                out = torch.irfft(out, 1, signal_sizes=(fft_size,))
            if out.shape[-1] > out_size:
                out = out[..., :out_size]
            elif out.shape[-1] < out_size:
                out = torch_pad(out, (0, out_size - out.shape[-1]))

            """
            Sum up the elements from computed output maps for each input 
            channel. Each output map has as many channels as the number of 
            filters. Each filter contributes one channel for the output map. 
            """
            # Sum the input channels.
            out = torch.sum(input=out, dim=1)
            # `unsqueeze` the dimension for channels.
            out = torch.unsqueeze(input=out, dim=0)
            output[batch_idx] = out
            if self.bias is not None:
                # Add the bias term for each filter.
                # Bias has to be unsqueezed to the dimension of the out to
                # properly sum up the values.
                output[batch_idx] += self.bias.unsqueeze(1)

        return output


class Conv1dfftCompressSignalOnly(Conv1dfftAutograd):

    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=None,
                 stride=None, padding=None, bias=True, index_back=None,
                 preserve_energy=100, out_size=None, filter_value=None,
                 bias_value=None, conv_index=None, use_next_power2=False,
                 is_complex_pad=True):
        super(Conv1dfftCompressSignalOnly, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias, index_back=index_back, preserve_energy=preserve_energy,
            out_size=out_size, filter_value=filter_value, bias_value=bias_value,
            conv_index=conv_index, use_next_power2=use_next_power2,
            is_complex_pad=is_complex_pad)

    def forward(self, input):
        """
        This is a manual implementation of the forward pass with the
        pytroch autograd used for the backward pass. We only compress the input
        signal and then pad the filter to the new length of the input signal,
        without any compression of the filter.

        :param input: the input map (e.g., a 1D signal such as time-series
        data.)
        :return: the result of 1D convolution.

        >>> x = np.array([[[1., 2., 3., 1., 4., 5., 10.]]])
        >>> y = np.array([[[2., 1., 3.]]])
        >>> b = np.array([0.0])
        >>> # print("Get the expected results from numpy correlate.")
        >>> expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        >>> conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(
        ... result, np.array([[expected_result]]))

        >>> # test without compression
        >>> x = np.array([[[1., 2., 3.]]])
        >>> y = np.array([[[2., 1.]]])
        >>> b = np.array([0.0])
        >>> conv_param = {'pad' : 0, 'stride' :1}
        >>> dout = np.array([[[0.1, -0.2]]])
        >>> # First, get the expected results from the numpy
        >>> # correlate function.
        >>> expected_result = np.correlate(x[0, 0,:], y[0, 0,:],
        ... mode="valid")
        >>> conv = Conv1dfftSimpleForLoop(filter_value=torch.from_numpy(y),
        ... bias_value=torch.from_numpy(b))
        >>> result = conv.forward(input=torch.from_numpy(x))
        >>> np.testing.assert_array_almost_equal(result,
        ... np.array([[expected_result]]))

        """
        batch_num = input.shape[0]
        filter_num = self.filter.shape[0]
        input_size = input.shape[-1]
        filter_size = self.filter.shape[-1]
        # Pad the input with the filter size (-1) for correctness.
        fft_size = input_size + self.kernel_size - 1
        if self.use_next_power2 is True:
            fft_size = next_power2(fft_size)
        input = torch_pad(input, (0, fft_size - input_size))
        # The input after fft is roughly 2 times smaller (in complex
        # representation) in terms of the length of the signal in the frequency
        # domain.
        input = torch.rfft(input, 1)
        init_half_fft_size = input.shape[-2]

        out_size = input_size - self.kernel_size + 1

        # Change from the percentage of how many coefficient should be
        # discarded to the the actual number of coefficients to discard.
        if self.index_back is not None and self.index_back > 0:
            if self.preserve_energy is not None and self.preserve_energy < 100:
                raise AttributeError(
                    "Choose either preserve_energy or index_back")
            # The index back has to be at least one coefficient removed
            # (thus: + 1)
            self.index_back = int(
                input.shape[-2] * (self.index_back / 100)) + 1

        if self.preserve_energy is not None and self.preserve_energy < 100:
            self.index_back = preserve_energy_index_back(
                input, self.preserve_energy)

        if self.index_back is not None and self.index_back > 0:
            # At least as many frequency coefficient has to remain as the filter
            # size.
            self.index_back = min(input.shape[-2] - filter_size,
                                  self.index_back)
            # This is complex representation in the last dimension, so the
            # length of the signal is the last but one dimension.
            full_energy, _ = get_full_energy_simple(x=input)
            input = input[..., :-self.index_back, :]
            preserved_energy, _ = get_full_energy_simple(x=input)

            # Estimate the size of the initial signal before fft if the
            # outcome of the fft would be the current input value.
            # We need the size of fft be at least the filter size.
            fft_size = max((input.shape[-2] - 1) * 2, filter_size)
            percent_retained_signal = 100 * (
                    init_half_fft_size - self.index_back) / init_half_fft_size
            percent_preserved_energy = preserved_energy / full_energy * 100
            msg = "conv_name," + "conv" + str(
                self.conv_index) + ",index_back," + str(
                self.index_back) + ",fft_size," + str(
                fft_size) + ",raw signal length," + str(
                input_size) + ",init_half_fft_size," + str(
                init_half_fft_size) + ",percent of preserved energy," + str(
                percent_preserved_energy) + (
                      ",percent of retained signal,") + str(
                percent_retained_signal)
            print(msg)
            with open(additional_log_file, "a") as file:
                file.write(msg + "\n")

        # Pad and transform the filters - after fft_size was decreased for the
        # input signal via compression.
        filter = torch_pad(self.filter, (0, fft_size - filter_size))
        filter = torch.rfft(filter, 1)

        output = torch.zeros([batch_num, filter_num, out_size],
                             dtype=input.dtype, device=input.device)

        for batch_idx in range(batch_num):
            # Broadcast each 1D signal with all filters.
            signal = input[batch_idx].unsqueeze(0)
            out = fast_jmul(signal, conj(filter))
            if out.shape[-1] < fft_size:
                out = complex_pad_simple(out, fft_size)
            out = torch.irfft(out, 1, signal_sizes=(fft_size,))
            if out.shape[-1] > out_size:
                out = out[..., :out_size]
            elif out.shape[-1] < out_size:
                out = torch_pad(out, (0, out_size - out.shape[-1]))

            """
            Sum up the elements from computed output maps for each input 
            channel. Each output map has as many channels as the number of 
            filters. Each filter contributes one channel for the output map. 
            """
            # Sum the input channels.
            out = torch.sum(input=out, dim=1)
            # `unsqueeze` the dimension for channels.
            out = torch.unsqueeze(input=out, dim=0)
            output[batch_idx] = out
            if self.bias is not None:
                # Add the bias term for each filter.
                # Bias has to be unsqueezed to the dimension of the out to
                # properly sum up the values.
                output[batch_idx] += self.bias.unsqueeze(1)

        return output


if __name__ == "__main__":
    test_run()
    torch.manual_seed(37)
    np.random.seed(13)
    import doctest

    sys.exit(doctest.testmod()[0])
