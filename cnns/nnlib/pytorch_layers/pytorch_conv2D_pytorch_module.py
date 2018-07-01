"""
Leverage the autograd in Pytorch.
"""

import logging
import torch
from torch.nn import Module
from torch.nn.functional import conv2d  # conv2d is actual cross-correlation in PyTorch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]


class PyTorchConv2d(Module):
    def __init__(self, filter_width, filter_height, filter=None, bias=None):
        super(PyTorchConv2d, self).__init__()
        if filter is None:
            self.filter = Parameter(torch.randn(1, 1, filter_width, filter_height))
        else:
            self.filter = filter
        if bias is None:
            self.bias = Parameter(torch.randn(1))
        else:
            self.bias = bias

    def forward(self, input):
        result = conv2d(input, self.filter)
        result += self.bias
        return result


if __name__ == "__main__":
    torch.manual_seed(231)
    module = PyTorchConv2d(3, 3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8, 8))
    print("gradient for the input: ", input.grad)
