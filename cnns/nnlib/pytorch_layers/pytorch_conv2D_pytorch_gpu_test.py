import torch
import unittest
from torch.autograd.gradcheck import gradcheck
from torch.nn.parameter import Parameter

from nnlib.pytorch_layers.pytorch_conv2D_pytorch_gpu import PyTorchConv2d


class TestPytorchConv2DPytorchSimple(unittest.TestCase):
    def test_scipy_conv2d_pytorch(self):
        # check the gradient
        # gradcheck takes a tuple of tensors as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all fulfill this condition
        moduleConv = PyTorchConv2d(3, 3)
        input = [torch.randn(1, 1, 20, 20, requires_grad=True)]
        test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
        print("Are the gradients correct: ", test)
        self.assertTrue(test)

    def test_simple(self):
        """
        Check the output of convolution and the computed gradient.
        """
        module = PyTorchConv2d(3, 3, filter=Parameter(torch.tensor([[[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]]])),
                               bias=Parameter(torch.tensor([0.0])))
        print("filter and bias parameters: ", list(module.parameters()))
        # input = torch.arange(end=16, dtype=torch.int32, requires_grad=True).view(4, 4)
        input = torch.tensor([[[[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], requires_grad=True)
        output = module(input)
        print("forward output: ", output)
        expected_output = torch.tensor([[[[348.0, 393], [528, 573]]]])
        self.assertTrue(output.equal(expected_output))
        output.backward(torch.tensor([[[[1.0, 2], [3, 4]]]]))
        print("gradient for the input: ", input.grad)
        # self.assertTrue(input.grad.equal(torch.tensor([[[[1.0, 4, 7, 6],
        #                                                  [7, 23, 33, 24],
        #                                                  [19, 53, 63, 42],
        #                                                  [21, 52, 59, 36]]]])))

        self.assertTrue(input.grad.equal(torch.tensor([[[[0., 0., 0., 0.],
                                                         [0., 1., 2., 0.],
                                                         [0., 3., 4., 0.],
                                                         [0., 0., 0., 0.]]]])))


if __name__ == '__main__':
    unittest.main()
