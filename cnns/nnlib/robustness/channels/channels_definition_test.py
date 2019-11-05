import logging
import unittest
import time
import numpy as np
import torch
from torch import tensor
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.utils.arguments import Arguments
from numpy.testing.utils import assert_allclose
from cnns.nnlib.robustness.channels.channels_definition import \
    svd_transformation


class TestChannelsDefinition(unittest.TestCase):

    def setUp(self):
        print("\n")
        log_file = "channels_definition_test.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        seed = 31
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("cuda is available")
            torch.cuda.manual_seed_all(seed)
        else:
            self.device = torch.device("cpu")
            print("cuda is not available")
            torch.manual_seed(seed)
        self.dtype = torch.float
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."
        self.args = Arguments()

    def testZeroOutLowFrequencyLshaped(self):
        a = torch.arange(40).reshape(8, 5)
        print('full a: ', a)
        n = 3
        a[n + 1:-n, :] = 0
        a[:, n + 1:] = 0
        print('zero-out a: ', a)

    def testSVDnumpyReconstructTorch(self):
        a = np.arange(9).reshape(1, 3, 3).astype(np.float)
        print('a: ', a)
        a_svd = svd_transformation(input=a, compress_rate=101)
        u = a_svd['u']
        s = a_svd['s']
        v = a_svd['v']

        u = u.transpose(1, 0)
        s = s.transpose(1, 0)

        print('u shape: ', u.shape)
        print('s shape: ', s.shape)
        print('v shape: ', v.shape)

        print('u: ', u)
        print('s: ', s)
        u_s = u * s
        print('u_s: ', u_s)

        x = u_s.matmul(v)
        x = x.unsqueeze(0)
        x = x.numpy()

        print('x: ', x)
        assert_allclose(actual=x, desired=a, rtol=1e-6, atol=1e-12)

    def testSVDnumpyReconstructTorchBatchMatrixMultiply(self):
        a = np.arange(9).reshape(1, 3, 3).astype(np.float)
        print('a: ', a)
        a_svd = svd_transformation(input=a, compress_rate=101)
        u = a_svd['u']
        s = a_svd['s']
        v = a_svd['v']

        print('u shape: ', u.shape)
        print('s shape: ', s.shape)
        print('v shape: ', v.shape)

        print('u: ', u)
        print('s: ', s)
        u_s = u * s
        print('u_s: ', u_s)

        u_s = u_s.unsqueeze(-1)
        v = v.unsqueeze(1)

        # Batch Matrix Multiply
        x = u_s.bmm(v)
        x = torch.sum(x, dim=0, keepdim=True)

        print('x: ', x)
        assert_allclose(actual=x, desired=a, rtol=1e-6, atol=1e-12)

    def testSVDnumpyReconstructTorchBatchMatrixMultiplyManyChannels(self):
        a = np.arange(27).reshape(3, 3, 3).astype(np.float)
        print('a: ', a)
        a_svd = svd_transformation(input=a, compress_rate=101)
        u = a_svd['u']
        s = a_svd['s']
        v = a_svd['v']

        print('u shape: ', u.shape)
        print('s shape: ', s.shape)
        print('v shape: ', v.shape)

        print('u: ', u)
        print('s: ', s)
        u_s = u * s
        print('u_s: ', u_s)

        u_s = u_s.unsqueeze(-1)
        v = v.unsqueeze(-2)

        # Batch Matrix Multiply
        x = u_s.matmul(v)
        x_0channel = torch.sum(x[:3], dim=0, keepdim=False)
        assert_allclose(actual=x_0channel, desired=a[0], atol=1e-12)
        x = torch.sum(x, dim=0, keepdim=True)

        print('x: ', x)
        desired = np.sum(a, axis=0, keepdims=True)
        assert_allclose(actual=x, desired=desired, rtol=1e-6, atol=1e-12)

    def testSVDnumpyReconstructTorchBatchMatrixMultiplyManyChannelsMatmul(self):
        a = np.arange(27).reshape(3, 3, 3).astype(np.float)
        print('a: ', a)
        a_svd = svd_transformation(input=a, compress_rate=101)
        u = a_svd['u']
        s = a_svd['s']
        v = a_svd['v']

        print('u shape: ', u.shape)
        print('s shape: ', s.shape)
        print('v shape: ', v.shape)

        print('u: ', u)
        print('s: ', s)

        u = u.transpose(1, 0)
        s = s.transpose(1, 0)

        u_s = u * s
        print('u_s: ', u_s)

        # Batch Matrix Multiply
        x = u_s.matmul(v)
        x = x.unsqueeze(0)
        print('x: ', x)
        desired = np.sum(a, axis=0, keepdims=True)
        assert_allclose(actual=x, desired=desired, rtol=1e-6, atol=1e-12)
