import torch
from torch import tensor
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfft
from cnns.nnlib.utils.arguments import Arguments
from torch.autograd import Function

import band_cuda

torch.manual_seed(31)


class BandFunction(Function):
    @staticmethod
    def forward(ctx, input, weights):
        out, xfft, yfft = band_cuda.forward(input, weights)
        ctx.save_for_backward(xfft, yfft)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad_x, grad_y = band_cuda.backward(
            grad.contiguous(), *ctx.saved_variables)
        return grad_x, grad_y


class Conv2dfftCpp(Conv2dfft):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=None, groups=None, bias=False,
                 weight_value=None, bias_value=None, is_manual=tensor([0]),
                 args=Arguments(), out_size=None):
        super(Conv2dfft, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
            weight_value=weight_value, bias_value=bias_value,
            is_manual=is_manual, args=args, out_size=out_size)

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
        return BandFunction.apply(input, self.weight)

