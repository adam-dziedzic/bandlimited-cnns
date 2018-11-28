from torch.autograd import Function
from torch import tensor

class ManipulateGradient(Function):

    def forward(self, input):
        # self.mark_dirty(input)
        print("forward Function")
        return input

    def backward(grad_out):
        # manipulate gradient here
        print("backward Function")
        return grad_out + 0.42

if __name__ == "__main__":
    layer = ManipulateGradient()
    input = tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = layer.apply(input)
    print("result: ", result)
    dout = tensor([0.1, -0.1, 0.2])
    result.backward(dout)
    print("result.grad: ", input.grad)