"""
Implement the forward and backward passes for convolution using scipy and numpy libraries. It requires us to go back and
forth between the tensors in numpy and tensors in PyTorch, which is not efficient.
"""
import numpy as np
import torch
from numpy import flip
from scipy.signal import correlate2d, convolve2d
from torch.autograd import Function
from torch.nn import Module
from torch.nn.parameter import Parameter

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]

class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        # grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        flipped_filter = flip(flip(filter.numpy(), axis=0), axis=1)
        # logger.debug("flipped filter: " + str(flipped_filter))
        grad_input = correlate2d(grad_output, flipped_filter, mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter), torch.from_numpy(grad_bias)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height, filter=None, bias=None):
        super(ScipyConv2d, self).__init__()
        if filter is None:
            self.filter = Parameter(torch.randn(filter_width, filter_height))
        else:
            self.filter = filter
        if bias is None:
            self.bias = Parameter(torch.randn(1, 1))
        else:
            self.bias = bias

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


if __name__ == "__main__":
    torch.manual_seed(231)
    module = ScipyConv2d(3, 3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(8, 8))
    print("gradient for the input: ", input.grad)
