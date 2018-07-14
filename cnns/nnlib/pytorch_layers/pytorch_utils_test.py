import torch
import unittest
from cnns.nnlib.pytorch_layers.pytorch_utils import flip


class TestPytorchUtils(unittest.TestCase):

    def test_flip(self):
        # check the flipping
        a = torch.tensor([[1, 2], [3, 4]])
        print("tensor a: ", a)
        print("flip(a, 0): ", flip(a, 0))
        flipped = flip(a, 0)
        expected = torch.tensor([[3, 4], [1, 2]])
        self.assertTrue(flipped.equal(expected))

    def test_dims_flip(self):
        b = torch.tensor([[[[1, 2], [3, 4]]]])
        print("tensor b: ", b)
        print("size of b: ", b.size())
        for dim in range(4):
            print("flip(b, {}) = ".format(dim), flip(b, dim))
        self.assertTrue(flip(b, 0).equal(b))
        self.assertTrue(flip(b, 1).equal(b))
        self.assertTrue(flip(b, 2).equal(torch.tensor([[[[3, 4], [1, 2]]]])))
        self.assertTrue(flip(b, 3).equal(torch.tensor([[[[2, 1], [4, 3]]]])))


    def test_double_flipped(self):
        b = torch.tensor([[[[1, 2], [3, 4]]]])
        double_flipped = flip(flip(b, 2), 3)
        print("flip(flip(b, 2), 3): ", double_flipped)
        double_expected = torch.tensor([[[[4, 3], [2, 1]]]])
        self.assertTrue(double_flipped.equal(double_expected))


if __name__ == '__main__':
    unittest.main()
