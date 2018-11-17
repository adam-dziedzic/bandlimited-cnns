import torch
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import NextPower2
import torch.nn as nn
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfft
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfft
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftAutograd
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfftAutograd
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftCompressSignalOnly
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftSimple
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfftSimpleForLoop


CONV_TYPE_ERROR = "Unknown type of convolution."

class Conv(object):

    def __init__(self, kernel_sizes, in_channels, out_channels, strides,
                 padding, args, is_debug=True, dtype=None, preserve_energy=100):
        """
        Create the convolution object from which we fetch the convolution
        operations.

        :param kernel_sizes: the sizes of the kernels in each conv layer.
        :param in_channels: number of channels in the input data.
        :param out_channels: the number of filters for each conv layer.
        :param strides: the strides for the convolutions.
        :param padding: padding for each convolutional layer.
        :param args: the general arguments for the program, e.g. the type of
        convolution to be used.
        :param is_debug: is the debug mode execution?
        :param dtype: the type of tensors.
        :param preserve_energy: how much energy to save in the input activation
        maps.
        """
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.padding = padding
        self.conv_type = ConvType[args.conv_type]
        self.index_back = args.index_back
        self.preserve_energy = preserve_energy
        self.is_debug = is_debug
        self.compress_type = CompressType[args.compress_type]
        self.args = args

        next_power2 = NextPower2[args.next_power2]
        self.next_power2 = True if next_power2 is NextPower2.TRUE else False

        self.dtype = dtype
        if self.dtype is None:
            tensor_type = TensorType[args.tensor_type]
            if tensor_type is TensorType.FLOAT32:
                self.dtype = torch.float32
            elif tensor_type is TensorType.FLOAT16:
                self.dtype = torch.float16
            elif tensor_type is TensorType.DOUBLE:
                self.dtype = torch.double
            else:
                raise ValueError(f"Unknown dtype: {tensor_type}")
            self.dtype = dtype

    def get_conv(self, index=0, index_back=None):
        if index == 0:
            in_channels = self.in_channels
        else:
            in_channels = self.out_channels[index - 1]

        if index_back is None:
            index_back = self.index_back

        if self.conv_type is ConvType.STANDARD:
            return nn.Conv1d(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=self.padding[index])
        elif self.conv_type is ConvType.STANDARD2D:
            return nn.Conv2d(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=self.padding[index])
        elif self.conv_type is ConvType.FFT1D:
            return Conv1dfft(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=self.padding[index],
                             index_back=index_back,
                             use_next_power2=self.next_power2,
                             conv_index=index,
                             preserve_energy=self.preserve_energy,
                             is_debug=self.is_debug,
                             compress_type=self.compress_type,
                             dtype=self.dtype)
        elif self.conv_type is ConvType.FFT2D:
            return Conv2dfft(in_channels=in_channels,
                             out_channels=self.out_channels[index],
                             stride=self.strides[index],
                             kernel_size=self.kernel_sizes[index],
                             padding=self.padding[index],
                             index_back=index_back,
                             use_next_power2=self.next_power2,
                             conv_index=index,
                             preserve_energy=self.preserve_energy,
                             is_debug=self.is_debug,
                             compress_type=self.compress_type,
                             dtype=self.dtype)
        elif self.conv_type is ConvType.AUTOGRAD:
            return Conv1dfftAutograd(in_channels=in_channels,
                                     out_channels=self.out_channels[index],
                                     stride=self.strides[index],
                                     kernel_size=self.kernel_sizes[index],
                                     padding=self.padding[index],
                                     index_back=self.index_back)
        elif self.conv_type is ConvType.AUTOGRAD2D:
            return Conv2dfftAutograd(in_channels=in_channels,
                                     out_channels=self.out_channels[index],
                                     stride=self.strides[index],
                                     kernel_size=self.kernel_sizes[index],
                                     padding=self.padding[index],
                                     index_back=self.index_back)
        elif self.conv_type is ConvType.SIMPLE_FFT:
            return Conv1dfftSimple(in_channels=in_channels,
                                   out_channels=self.out_channels[index],
                                   stride=self.strides[index],
                                   kernel_size=self.kernel_sizes[index],
                                   padding=self.padding[index],
                                   index_back=self.index_back)
        elif self.conv_type is ConvType.SIMPLE_FFT_FOR_LOOP:
            return Conv1dfftSimpleForLoop(in_channels=in_channels,
                                          out_channels=self.out_channels[index],
                                          stride=self.strides[index],
                                          kernel_size=self.kernel_sizes[index],
                                          padding=self.padding[index],
                                          index_back=self.index_back)
        elif self.conv_type is ConvType.COMPRESS_INPUT_ONLY:
            return Conv1dfftCompressSignalOnly(
                in_channels=in_channels, out_channels=self.out_channels[index],
                stride=self.strides[index],
                kernel_size=self.kernel_sizes[index],
                padding=self.padding[index],
                index_back=self.index_back,
                preserve_energy=self.preserve_energy)
        else:
            raise CONV_TYPE_ERROR