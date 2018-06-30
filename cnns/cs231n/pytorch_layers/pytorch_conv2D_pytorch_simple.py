import torch
from torch.autograd import Function
from torch.nn.functional import conv2d  # conv2d is actual cross-correlation in PyTorch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def flip(x, dim):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension acco
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
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
        grad_input = conv2d(grad_output, torch.flip(torch.flip(filter.numpy(), dim=0), dim=1), mode='full')
        grad_filter = conv2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter), torch.from_numpy(grad_bias)


class PyTorchConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(PyTorchConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return PyTorchConv2dFunction.apply(input, self.filter, self.bias)


if __name__ == "__main__":
    from torch.autograd.gradcheck import gradcheck

    module = PyTorchConv2d(3, 3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(10, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(8, 8))
    print("gradient for the input: ", input.grad)

    # check the gradient
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all fulfill this condition

    moduleConv = PyTorchConv2d(3, 3)
    input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    print("Are the gradients correct: ", test)
