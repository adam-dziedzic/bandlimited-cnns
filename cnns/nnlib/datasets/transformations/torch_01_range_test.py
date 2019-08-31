import unittest
import logging
import torch
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.datasets.cifar10_example import cifar10_example
from .torch_01_range import Ranger
from cnns.nnlib.attacks.simple_blackbox_remote.utils import apply_normalization
from cnns.nnlib.attacks.simple_blackbox_remote.utils import invert_normalization

class TestTorch01Range(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv1D_reuse_map_fft.log"
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

    def test_cifar_to_01(self):
        dataset = 'cifar'
        ranger = Ranger(device=self.device)
        x = torch.tensor(cifar10_example).to(self.device)
        y = ranger.to_01(x, dataset=dataset)
        print(y.min(), y.max())
        # assert y.min() >= 0.0
        assert torch.allclose(torch.min(torch.tensor([y.min().item(), 0.0])),
                              torch.tensor([0.0]), atol=1e-05), f'{y.min()} is substantially lower than 0.0'
        # assert y.max() <= 1.0
        assert torch.allclose(torch.max(torch.tensor([y.max().item(), 1.0])),
                              torch.tensor([1.0])), f'{y.max()} is substantially greater than 1.0'
        y2 = invert_normalization(x, dataset='cifar')
        assert torch.allclose(y, y2)

    def test_cifar_to_torch(self):
        dataset = 'cifar'
        x = torch.tensor(cifar10_example).to(self.device)
        ranger = Ranger(device=self.device)
        y_01 = ranger.to_01(x, dataset=dataset)
        y2_01 = invert_normalization(x, dataset=dataset)
        y_torch = ranger.to_torch(y_01, dataset=dataset)
        y2_torch = apply_normalization(y2_01, dataset=dataset)
        assert torch.allclose(x, y_torch)
        assert torch.allclose(y_torch, y2_torch)

