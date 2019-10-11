import logging
import sys
import math
import numpy as np
import torch
from torch_dct import dct, idct
import time
from torch import tensor
from torch.nn import Module
from torch.nn.functional import pad as torch_pad
from torch.nn.parameter import Parameter
from torch.nn import init

from cnns.nnlib.pytorch_layers.pytorch_utils import correlate_fft_signals2D
from cnns.nnlib.pytorch_layers.pytorch_utils import from_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import get_pair
from cnns.nnlib.pytorch_layers.pytorch_utils import preserve_energy2D_symmetry
from cnns.nnlib.pytorch_layers.pytorch_utils import pytorch_conjugate
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul2
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul3
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul4
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul5
from cnns.nnlib.pytorch_layers.pytorch_utils import to_tensor
from cnns.nnlib.pytorch_layers.pytorch_utils import cuda_mem_show
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_index_forward
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_index_back
from cnns.nnlib.pytorch_layers.pytorch_utils import zero_out_min
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_elem_size
from cnns.nnlib.pytorch_layers.pytorch_utils import get_tensors_elem_size
from cnns.nnlib.pytorch_layers.pytorch_utils import get_step_estimate
from cnns.nnlib.pytorch_layers.pytorch_utils import restore_size_2D
from cnns.nnlib.utils.general_utils import CompressType, next_power2
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.utils.general_utils import additional_log_file


class ConvDCT(Module):
    """
    :conv_index_counter: the counter to index (number) of the convolutional
    2d fft layers .
    """
    conv_index_counter = 0

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=None, groups=None, bias=False,
                 weight_value=None, bias_value=None, is_manual=tensor([0]),
                 args=Arguments(), out_size=None):
        """

        2D convolution using DCT.

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
        :param compress_rate: how many frequency coefficients should be
        discarded
        :param preserve_energy: how much energy should be preserved in the input
        image.
        :param out_size: what is the expected output size of the
        operation (when compression is used and the out_size is
        smaller than the size of the input to the convolution, then
        the max pooling can be omitted and the compression
        in this layer can serve as the frequency-based (spectral)
        pooling.
        :param weight_value: you can provide the initial filter, i.e.
        filter weights of shape (F, C, HH, WW), where
        F - number of filters, C - number of channels, HH - height of the
        filter, WW - width of the filter
        :param bias_value: you can provide the initial value of the bias,
        of shape (F,)
        :param use_next_power2: should we extend the size of the input for the
        FFT convolution to the next power of 2.
        :param is_manual: to check if the backward computation of convolution
        was computed manually.

        Regarding the stride parameter: the number of pixels between
        adjacent receptive fields in the horizontal and vertical
        directions, we can generate the full output, and then remove the
        redundant elements according to the stride parameter. The more relevant
        method is to apply spectral pooling as a means to achieving the strided
        convolution.
        """
        super(ConvDCT, self).__init__()
        self.args = args

        if dilation is not None and dilation > 1:
            raise NotImplementedError("dilation > 1 is not supported.")
        if groups is not None and groups > 1:
            raise NotImplementedError("groups > 1 is not supported.")

        self.is_weight_value = None  # Was the filter value provided?
        if weight_value is None:
            self.is_weight_value = False
            if out_channels is None or in_channels is None or \
                    kernel_size is None:
                raise ValueError("Either specify filter_value or provide all"
                                 "the required parameters (out_channels, "
                                 "in_channels and kernel_size) to generate the "
                                 "filter.")
            self.kernel_height, self.kernel_width = get_pair(kernel_size)
            if args.dtype is torch.float:
                weight = torch.randn(out_channels, in_channels,
                                     self.kernel_height,
                                     self.kernel_width, dtype=args.dtype)
            elif args.dtype is torch.half:
                weight = torch.randn(out_channels, in_channels,
                                     self.kernel_height,
                                     self.kernel_width).to(torch.half)
            else:
                raise Exception(f"Unknown dtype in args: {args.dtype}")
            self.weight = Parameter(weight)
        else:
            self.is_weight_value = True
            self.weight = weight_value
            out_channels = weight_value.shape[0]
            in_channels = weight_value.shape[1]
            self.kernel_height = weight_value.shape[2]
            self.kernel_width = weight_value.shape[3]

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

        self.pad_H, self.pad_W = get_pair(value=padding, val_1_default=0,
                                          val2_default=0,
                                          name="padding")

        if self.pad_H != self.pad_W:
            raise Exception(
                "We only support a symmetric padding in the frequency domain.")

        self.stride = stride
        self.stride_type = args.stride_type

        self.stride_H, self.stride_W = get_pair(value=stride,
                                                val_1_default=None,
                                                val2_default=None,
                                                name="stride")

        if self.stride_H != self.stride_W:
            raise Exception(
                "We only support a symmetric striding in the frequency domain.")

        self.is_manual = is_manual
        self.conv_index = ConvDCT.conv_index_counter
        ConvDCT.conv_index_counter += 1
        self.out_size = out_size

        self.out_size_H, self.out_size_W = get_pair(value=out_size,
                                                    val_1_default=None,
                                                    val2_default=None,
                                                    name="out_size")

        if self.out_size_H != self.out_size_W:
            raise Exception(
                "We only support a symmetric outputs in the frequency domain.")

        if args is None:
            self.compress_rate = None
            self.preserve_energy = None
            self.is_debug = False
            self.next_power2 = False
            self.is_debug = False
            self.compress_type = CompressType.STANDARD
        else:
            self.compress_rate = args.compress_rate
            self.preserve_energy = args.preserve_energy
            self.next_power2 = args.next_power2
            self.is_debug = args.is_debug
            self.compress_type = args.compress_type

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_weight_value is not None and self.is_weight_value is False:
            if self.weight.dtype is torch.half:
                dtype = self.weight.dtype
                weight = self.weight.to(torch.float)
                init.kaiming_uniform_(weight, a=math.sqrt(5))
                self.weight = Parameter(weight.to(dtype))
            else:
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None and self.is_bias_value is False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def out_HW(self, H, W, HH, WW):
        if self.out_size_H:
            out_H = self.out_size_H
        elif self.out_size or self.stride_type is StrideType.SPECTRAL:
            out_H = (H - HH + 2 * self.pad_H) // self.stride_H + 1
        else:
            out_H = H - HH + 1 + 2 * self.pad_H

        if self.out_size_W:
            out_W = self.out_size_W
        elif self.out_size or self.stride_type is StrideType.SPECTRAL:
            out_W = (W - WW + 2 * self.pad_W) // self.stride_W + 1
        else:
            out_W = W - WW + 1 + 2 * self.pad_W

        if out_H != out_W:
            raise Exception(
                "We only support a symmetric compression in the frequency domain.")
        return out_H, out_W

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
        filter = self.weight
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

        pad_filter_H = H - HH
        pad_filter_W = W - WW

        filter = torch_pad(
            filter, (0, pad_filter_W, 0, pad_filter_H), 'constant', 0)

        input = dct(input)
        filter = dct(filter)
        # permute from N, C, H, W to H, W, N, C
        input = input.permute(2, 3, 0, 1)
        # permute from F, C, H, W to H, W, C, F
        filter = filter.permute(2, 3, 1, 0)
        result = torch.matmul(input, filter)
        # permute from H, W, N, F to N, F, H, W
        result = result.permute(2, 3, 0, 1)
        result = idct(result)
        out_H, out_W = self.out_HW(H, W, HH, WW)
        result = result[..., :out_H, :out_W]
        if self.bias is not None:
            # Add the bias term for each filter (it has to be unsqueezed to
            # the dimension of the out to properly sum up the values).
            unsqueezed_bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            result += unsqueezed_bias
        if (self.stride_H != 1 or self.stride_W != 1) and (
                self.stride_type is StrideType.STANDARD):
            result = result[:, :, ::self.stride_H, ::self.stride_W]
        return result
