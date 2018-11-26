import unittest
import logging
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.datasets.ucr.dataset import UCRDataset
from cnns.nnlib.datasets.ucr.dataset import ToTensor
from cnns.nnlib.datasets.ucr.dataset import AddChannel
from cnns.nnlib.datasets.ucr.ucr import get_dev_dataset
import torch
from torchvision import transforms
from cnns.nnlib.utils.arguments import Arguments


class TestUCRdataset(unittest.TestCase):

    def setUp(self):
        log_file = "tsetUCRdataset.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def testDevDataset(self):
        args = Arguments()
        args.dev_percent = 30
        args.dataset_name = "50words"
        train_dataset = UCRDataset(args.dataset_name, train=True,
                                   transformations=transforms.Compose(
                                       [ToTensor(dtype=torch.float),
                                        AddChannel()]))
        print("length of the train dataset: ", len(train_dataset))
        self.assertEqual(len(train_dataset), 450)

        dev_dataset = get_dev_dataset(args=args, train_dataset=train_dataset)
        self.assertEqual(len(dev_dataset), 135)
        self.assertEqual(len(train_dataset), 315)
