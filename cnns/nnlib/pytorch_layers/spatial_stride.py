import torch
from torch.nn import Module
from torch import autograd
from torch import tensor


class SpatialStrideFunction(autograd.Function):
    """
    Corresponds to pooling with a 2x2 filter with stride 2 and retaining only
    the top-left cell in the input tile covered by the filter.
    """

    @staticmethod
    def forward(ctx, input, stride=2):
        """
        Pooling with a 2x2 filter with stride 2 and retaining only the top-left
        cell in the input tile covered by the filter

        :param ctx: function context.
        :param input: the 2D input with batch and channel dimensions.
        :return: top-left value
        """
        if ctx is not None:
            ctx.save_for_backward(input, tensor([stride]))
        return SpatialStrideFunction.topLeftPooling(input, stride)

    @staticmethod
    def topLeftPooling(input, stride):
        output = input[..., ::stride, ::stride]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the
        gradient of the loss with respect to the output, and we need
        to compute the gradient of the loss with respect to the input.
        """
        input, stride = ctx.saved_tensors
        stride = stride.item()
        grad_input = torch.zeros_like(input)
        grad_input[..., ::stride, ::stride] = grad_output
        return grad_input


class SpatialStride(Module):

    def __init__(self):
        super(SpatialStride, self).__init__()

    def forward(self, input):
        return SpatialStrideFunction.apply(input)


class SpatialStrideAutograd(SpatialStride):

    def __init__(self):
        super(SpatialStrideAutograd, self).__init__()

    def forward(self, input):
        return SpatialStrideFunction.forward(ctx=None, input=input)
