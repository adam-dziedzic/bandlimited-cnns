import torch
from torch.autograd import Function
from torch.nn.functional import conv2d  # conv2d is actual cross-correlation in PyTorch
from torch.nn import Module
from torch.nn.parameter import Parameter


def flip(x, dim):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class PyTorchConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input, filter, bias
        result = conv2d(input, filter)
        result += bias
        ctx.save_for_backward(input, filter, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, filter, bias = ctx.saved_tensors
        grad_bias = torch.sum(grad_output)
        flipped_filter = flip(flip(filter, dim=2), dim=3)
        grad_input = conv2d(grad_output, flipped_filter, padding=(filter.size(2) - 1, filter.size(3) - 1))
        grad_filter = conv2d(input, grad_output)
        return grad_input, grad_filter, grad_bias


class PyTorchConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(PyTorchConv2d, self).__init__()
        self.filter = Parameter(torch.randn(1, 1, filter_width, filter_height))
        self.bias = Parameter(torch.randn(1))

    def forward(self, input):
        return PyTorchConv2dFunction.apply(input, self.filter, self.bias)


if __name__ == "__main__":

    # check the flipping
    a = torch.tensor([[1, 2], [3, 4]])
    print("tensor a: ", a)
    print("flip(a, 0): ", flip(a, 0))

    b = torch.tensor([[[[1, 2], [3, 4]]]])
    print("tensor b: ", b)
    print("size of b: ", b.size())
    for dim in range(4):
        print("flip(b, {}".format(dim), flip(b, dim))

    print("flip(flip(b, 2), 3): ", flip(flip(b, 2), 3))

    from torch.autograd.gradcheck import gradcheck

    module = PyTorchConv2d(3, 3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8, 8))
    print("gradient for the input: ", input.grad)

    # check the gradient
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all fulfill this condition

    moduleConv = PyTorchConv2d(3, 3)
    input = [torch.randn(1, 1, 20, 20, requires_grad=True)]
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    print("Are the gradients correct: ", test)
