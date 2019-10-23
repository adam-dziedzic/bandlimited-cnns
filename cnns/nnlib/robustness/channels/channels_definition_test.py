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
        a[n+1:-n, :] = 0
        a[:, n+1:] = 0
        print('zero-out a: ', a)