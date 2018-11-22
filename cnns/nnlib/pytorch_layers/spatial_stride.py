import torch
from torch.nn import Module
from torch import autograd


class SpatialStrideFunction(autograd.Function):
    """
    Corresponds to pooling with a 2x2 filter with stride 2 and retaining only
    the top-left cell in the input tile covered by the filter.
    """

    @staticmethod
    def forward(ctx, input):
        """
        Pooling with a 2x2 filter with stride 2 and retaining only the top-left
        cell in the input tile covered by the filter

        :param ctx: function context.
        :param input: the 2D input with batch and channel dimensions.
        :return: top-left value
        """
        if ctx is not None:
            ctx.save_for_backward(input)
        return SpatialStrideFunction.topLeftPooling(input)

    @staticmethod
    def topLeftPooling(input):
        output = input[..., ::2, ::2]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the
        gradient of the loss with respect to the output, and we need
        to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input[..., ::2, ::2] = grad_output
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
