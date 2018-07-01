"""
All operations in PyTorch with manual autograd.
"""

import torch
from torch.autograd import Function
from torch.nn import Module
from torch.nn.functional import conv2d  # conv2d is actual cross-correlation in PyTorch
from torch.nn.parameter import Parameter

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]

class PyTorchConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        result = conv2d(input, filter)
        result += bias
        ctx.save_for_backward(input, filter, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, filter, bias = ctx.saved_tensors
        grad_bias = torch.tensor([torch.sum(grad_output)])
        flipped_filter = filter.flip(dims=[2,3])
        # logger.debug("flipped filter: " + str(flipped_filter))
        grad_input = conv2d(grad_output, flipped_filter, padding=(filter.size(2) // 2, filter.size(3) // 2))
        grad_filter = conv2d(input, grad_output)
        return grad_input, grad_filter, grad_bias


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
        return PyTorchConv2dFunction.apply(input, self.filter, self.bias)


if __name__ == "__main__":
    torch.manual_seed(231)
    module = PyTorchConv2d(3, 3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8, 8))
    print("gradient for the input: ", input.grad)

"""
expected gradient for the input:  tensor([[-0.2478, -0.7275, -1.0670,  1.3629,  1.7458, -0.5786, -0.2722,
          0.5767,  0.7379, -0.6335],
        [-1.7113,  0.1839,  1.0434, -3.5176, -1.7056,  1.0892,  2.0054,
          2.3190, -1.6143, -1.3427],
        [-2.4303, -0.1218,  1.9863, -1.6753, -0.3529, -2.4454,  0.4331,
          1.8996,  1.5348, -0.3813],
        [-1.7727, -1.4130,  2.8780, -0.1220, -1.1942,  0.9997, -2.8926,
         -1.4083,  1.1635,  0.9641],
        [ 0.2487,  0.0023,  0.3793, -0.4038,  1.3017,  0.1421, -0.9947,
          0.5084,  0.1511, -2.1860],
        [-0.1263,  1.7602,  3.3994,  0.7883,  0.6831, -0.7291, -0.3211,
          1.8856,  0.3729, -1.2780],
        [-2.1050,  1.8296,  2.4018,  0.5756,  1.3364, -2.9692, -0.4314,
          3.3727,  3.1612, -1.0387],
        [-0.5624, -1.0603,  0.8454,  0.2767,  0.3005,  0.3977, -1.1085,
         -2.7611, -0.4906, -0.1018],
        [ 0.4603, -0.7684,  1.0566, -0.8825,  0.8468,  1.0482,  1.2088,
          0.2836,  0.0993, -0.0322],
        [ 0.0131,  0.4351, -0.3529,  0.2088, -0.3471,  0.3255,  1.6812,
          0.1925, -0.6875,  0.1037]])
"""
