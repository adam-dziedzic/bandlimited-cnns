import torch
import unittest

from cnns.nnlib.pytorch_layers.MyReLU import MyReLU


class TestMyReLU(unittest.TestCase):

    def test_relu(self):
        dtype = torch.float
        device = torch.device("cpu")
        # uncomment the line below to run on gpu
        # device = torch.device("cuda:0")

        # To apply our Function, we use Function.apply method.
        # We alias this as 'relu'.
        relu = MyReLU.apply
        print("type of the relu: ", type(relu))
        a = torch.tensor([[1, -2], [-3, 4]], requires_grad=True,
                         dtype=dtype, device=device)
        print("a grad: ", a.grad)
        result = relu(a)
        expected = torch.tensor([[1, 0], [0, 4]], dtype=dtype,
                                device=device)
        self.assertTrue(result.equal(expected))

        gradienty_y = torch.tensor([[0.1, 0.1], [0.1, 0.1]],
                                   dtype=dtype, device=device)
        result.backward(gradienty_y)
        print("data a: ", a)
        print("a final grad: ", a.grad)
        expected_gradient = torch.tensor([[0.1, 0.], [0., 0.1]])
        self.assertTrue(a.grad.equal(expected_gradient))


if __name__ == '__main__':
    unittest.main()
