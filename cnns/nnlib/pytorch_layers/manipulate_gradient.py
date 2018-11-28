from torch.autograd import Function
from torch import tensor
import torch

class ManipulateGradient(Function):

    def forward(self, input):
        self.mark_dirty(input)
        print("forward Function")
        return input

    def backward(self, grad_out):
        # manipulate gradient here
        print("backward Function")
        return grad_out + 0.42

class ManipulateGradientStatic(Function):

    @staticmethod
    def forward(ctx, input):
        # ManipulateGradientStatic.mark_dirty(input)
        print("forward static")
        return input

    @staticmethod
    def backward(ctx, grad_out):
        # manipulate gradient here
        print("backward static")
        return grad_out + 0.42

class ManipulateGradientModule(torch.nn.modules.Module):
    def __init__(self):
        super(ManipulateGradientModule, self).__init__()

    def forward(self, input):
        print("forward Module")
        return ManipulateGradientStatic.apply(input)

if __name__ == "__main__":
    layer = ManipulateGradientModule()
    input = tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = layer.forward(input)
    print("result: ", result)
    dout = tensor([0.1, -0.1, 0.2])
    result.backward(dout)
    print("result.grad: ", input.grad)
