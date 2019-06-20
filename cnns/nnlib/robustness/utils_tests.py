import unittest
import numpy as np
import torch
from cnns.nnlib.robustness.utils import elem_wise_dist
from cnns.nnlib.robustness.utils import norm
from cnns.nnlib.robustness.utils import most_frequent_class
from cnns.nnlib.robustness.utils import elem_wise_dist
from numpy.testing import assert_equal
from numpy.testing import assert_allclose
from numpy import linalg as LA


class TestUtils(unittest.TestCase):

    def test_dist(self):
        t1 = np.array([[[1.0, 2, 3]]])
        t2_1 = [[[2.0, 2, 3]]]
        t2_2 = [[[1.0, 4, 3]]]
        t2 = np.array([t2_1, t2_2])
        dist_all = elem_wise_dist(t1, t2)
        # print("dist all: ", dist_all)
        assert_equal(dist_all, [1, 2])

    def test_dist2(self):
        a = np.array([[[[1.0, 2],
                        [3, 4]]],
                      [[[4.0, 5],
                        [6, 7]]]])
        b = np.array([[[[0.0, 2], [4, 4]]]])
        my_elem_wise_dists = elem_wise_dist(b, a)
        print("my_elem_wise_dists: ", my_elem_wise_dists)

        diff = torch.from_numpy(a) - torch.from_numpy(b)
        dims = (2, 3)
        torch_elem_wise_dists = torch.norm(diff, dim=dims)
        print("elem_wise_dists: ", torch_elem_wise_dists)
        print("avg. dist: ", np.average(torch_elem_wise_dists.numpy()))

        assert_allclose(actual=my_elem_wise_dists,
                        desired=torch_elem_wise_dists.flatten())

    def test_general(self):
        a = np.random.rand(4, 5, 5)
        axis = 1
        for p in [1, 2, np.inf]:
            assert_allclose(norm(a, p=p, axis=axis),
                            LA.norm(a, ord=p, axis=axis))

    def test1(self):
        b = np.array([[-4, -3, -2],
                      [-1, 0, 1],
                      [2, 3, 4]])
        # b = np.random.rand(4, 5, 5)
        print()
        axis=1
        for p in [1, 2, np.inf]:
            print("norm order: ", p)
            my_result = norm(b, axis=axis, p=p)
            print("my result: ", my_result)
            la_result = LA.norm(b, ord=p, axis=axis)
            print("la_result: ", la_result)
            assert_allclose(actual=my_result, desired=la_result)


    def test_most_frequent_class(self):
        predictions = np.array([[1]])
        class_id = most_frequent_class(predictions)
        assert_equal(actual=class_id, desired=0)

        predictions = np.array([[1,2,3,1,1]])
        class_id = most_frequent_class(predictions)
        assert_equal(actual=class_id, desired=2)

        predictions = np.array([[1,2,1,4,1],[5,4,1,2,0],[1,1,0,2,1]])
        class_id = most_frequent_class(predictions)
        assert_equal(actual=class_id, desired=3)

    def test_elem_wise_dist(self):
        a = [1.0, 2, 3, 4, 1]
        b = [3.0, 1, 2, 1, 5]
        aa = np.array(a)
        bb = np.array(b)
        dist_desire = np.linalg.norm(aa - bb, ord=2)
        aaa = np.array([[a]])
        bbb = np.array([[[b]]])
        dist_got = elem_wise_dist(aaa, bbb, p=2)
        assert_equal(actual=dist_got, desired=dist_desire)

if __name__ == '__main__':
    unittest.main()
