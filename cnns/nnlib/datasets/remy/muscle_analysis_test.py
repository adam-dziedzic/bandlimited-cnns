import unittest
import numpy as np
from numpy.testing import assert_allclose
from cnns.nnlib.datasets.remy.muscle_analysis import normalize_with_nans


class TestUtils(unittest.TestCase):

    def test_normalize_with_nans(self):
        a = np.arange(9).reshape(3, 3).astype(np.float)
        a[0][0] = 999
        a[1][1] = 999
        a[0][1] = 20
        a[1][0] = -2
        print("\na: ", a)
        b, _, _ = normalize_with_nans(a, nans=999)
        print("\b: ", b)
        c = np.array([[0.0, 1, -1.224745], [-1, 0, 0], [1, -1, 1.224745]])
        assert_allclose(actual=b, desired=c, rtol=1e-6)
