import torch
import unittest

from cnns.nnlib.pytorch_layers.spatial_stride import SpatialStride
from cnns.nnlib.pytorch_layers.spatial_stride import SpatialStrideAutograd
from torch.autograd.gradcheck import gradcheck
import numpy as np
from numpy import testing


class TestSpatialStride(unittest.TestCase):

    def test_spatial_stride(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        a = torch.tensor(
            [[[[1, -2, 3.0, 1, -2],
               [-3, 4, -1.0, 2, 5],
               [1.0, 9.0, 0.5, -1, -2],
               [3, -6, 1.0, -2, 7],
               [-1, -2, -3.0, -1, 2]]]],
            requires_grad=True, dtype=dtype, device=device)
        a_copy = a.clone().detach().requires_grad_(True)
        print("\n")
        print("data a: ", a)
        print("a grad: ", a.grad)

        spatialManualGrad = SpatialStride()
        resultManual = spatialManualGrad(a)

        spatialAutoGrad = SpatialStrideAutograd()
        resultAuto = spatialAutoGrad(a_copy)

        expected = torch.tensor([[[[1, 3.0, -2],
                                   [1.0, 0.5, -2],
                                   [-1, -3.0, 2]]]], dtype=dtype).numpy()

        MSG = "The expected 'desired' is different than the obtained 'actual'."

        testing.assert_allclose(actual=resultManual.detach().cpu().numpy(),
                                desired=expected,
                                err_msg=MSG)
        testing.assert_allclose(actual=resultAuto.detach().cpu().numpy(),
                                desired=expected,
                                err_msg=MSG)

        gradient_y = torch.tensor(
            [[[[0.1, -0.1, 0.2],
               [0.2, -0.2, 0.1],
               [0.3, -0.3, 2.0]]]],
            dtype=dtype, device=device)

        resultManual.backward(gradient_y)
        resultAuto.backward(gradient_y)

        print("a final grad: ", a.grad)
        print("a clone final grad: ", a_copy.grad)
        expected_gradient = torch.tensor(
            [[[[0.1, 0, -0.1, 0, 0.2],
               [0, 0, 0, 0, 0],
               [0.2, 0, -0.2, 0, 0.1],
               [0, 0, 0, 0, 0],
               [0.3, 0, -0.3, 0, 2.0]]]], dtype=dtype).numpy()
        testing.assert_allclose(actual=a.grad.cpu().numpy(),
                                desired=expected_gradient,
                                err_msg=MSG)
        testing.assert_allclose(actual=a_copy.grad.cpu().numpy(),
                                desired=expected_gradient,
                                err_msg=MSG)

        print("Manual gradient check passed.")

        c = a.clone().detach().requires_grad_(True)
        test = gradcheck(spatialManualGrad, inputs=[c], eps=1e-4, atol=1e-4)
        print("Are the gradients correct: ", test)
        self.assertTrue(test)


if __name__ == '__main__':
    unittest.main()
