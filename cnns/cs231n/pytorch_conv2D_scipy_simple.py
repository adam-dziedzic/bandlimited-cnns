import numpy as np
import
import torch
from numpy import flip
from torch.autograd import Function
from torch.autograd import gradcheck
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class NumpyConv1dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = np.correlate(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = np.correlate(grad_output, flip(filter.numpy(), axis=0), mode='full')
        grad_filter = np.correlate(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter), torch.from_numpy(grad_bias)


class NumpyConv1d(Module):
    def __init__(self, filter_width, filter_height):
        super(NumpyConv1d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width))
        self.bias = Parameter(torch.randn(1,1))

    def forward(self, input):
        return NumpyConv1dFunction.apply(input, self.filter, self.bias)


if __name__ == "__main__":
    module = NumpyConv1d(4)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(8))
    print("gradient for the input: ", input.grad)

    # check the gradient
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all fulfill this condition

    moduleConv = NumpyConv1d(5)

    input = [torch.randn(20, dtype=torch.double, requires_grad=True)]
    # print("input: ", input)
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    print("Are the gradients correct: ", test)
