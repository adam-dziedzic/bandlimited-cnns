import torch
import unittest

from cnns.nnlib.pytorch_layers.round import Round
from cnns.nnlib.utils.arguments import Arguments
import numpy as np

class TestRound(unittest.TestCase):

    def test_round(self):
        # uncomment the line below to run on gpu
        # device = torch.device("conv1D_cuda:0")

        # To apply our Function, we use Function.apply method.
        # We alias this as 'relu'.
        args = Arguments()
        args.mean_array = np.array((0.0, 0.0), dtype=np.float32).reshape((2, 1))
        args.std_array = np.array((1.0, 1.0), dtype=np.float32).reshape((2, 1))
        args.dtype = torch.float
        args.device = torch.device("cpu")
        args.values_per_channel = 2

        round = Round(args=args)
        a = torch.tensor([[0.6, 0.2], [0.1, 0.9]], requires_grad=True,
                         dtype=args.dtype, device=args.device)
        print("a grad: ", a.grad)
        result = round(a)
        print("result of the forward pass: ", result)
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=args.dtype,
                                device=args.device)
        self.assertTrue(result.equal(expected))

        gradienty_y = torch.tensor([[0.1, 0.2], [0.4, 0.3]],
                                   dtype=args.dtype, device=args.device)
        result.backward(gradienty_y)
        print("data a: ", a)
        print("a final grad: ", a.grad)
        expected_gradient = torch.tensor([[0.1, 0.2], [0.4, 0.3]])
        self.assertTrue(a.grad.equal(expected_gradient))


if __name__ == '__main__':
    unittest.main()
