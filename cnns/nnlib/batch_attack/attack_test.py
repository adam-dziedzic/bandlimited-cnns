import unittest
import torch

from cnns.nnlib.robustness.batch_attack.attack import linf_batch
from cnns.nnlib.robustness.batch_attack.attack import linf_torch
from cnns.nnlib.robustness.batch_attack.attack import linf_for

from cnns.nnlib.robustness.batch_attack.attack import l2_batch
from cnns.nnlib.robustness.batch_attack.attack import l2_torch
from cnns.nnlib.robustness.batch_attack.attack import l2_for


class DistanceTestCase(unittest.TestCase):

    def test_linf_batch(self):
        a = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8.0]],
                          [[2, 9],
                           [-19, 0]]])
        self.assertEqual(linf_batch(a), 31)
        self.assertEqual(linf_batch(a), linf_for(a))
        self.assertEqual(linf_batch(a), linf_torch(a))

        b = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8.0]]])
        linf_sum = linf_batch(b)
        self.assertEqual(linf_sum, 12.0)
        self.assertEqual(linf_sum, linf_torch(b))
        self.assertEqual(linf_sum, linf_for(b))

        c = torch.randn(2, 3, 4)
        self.assertEqual(linf_batch(c), linf_for(c))
        self.assertEqual(linf_batch(c), linf_torch(c))

    def test_l2_batch(self):
        a = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8.0]],
                          [[2, 9],
                           [-19, 0]]])
        self.assertEqual(l2_batch(a), l2_for(a))
        self.assertEqual(l2_batch(a), l2_torch(a))

        b = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8.0]]])
        self.assertEqual(l2_batch(b), l2_torch(b))
        self.assertEqual(l2_batch(b), l2_for(b))

        c = torch.randn(2, 3, 4)
        self.assertEqual(l2_batch(c), l2_for(c))
        self.assertEqual(l2_batch(c), l2_torch(c))


if __name__ == '__main__':
    unittest.main()
