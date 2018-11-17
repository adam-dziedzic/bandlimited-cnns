import torch.nn as nn
import torch
import torch.nn.functional as F
from cnns.nnlib.pytorch_layers.conv_picker import Conv

class LeNet(nn.Module):
    """
    Based on:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, input_size, args, num_classes=10, in_channels=1,
                 dtype=torch.float32, kernel_sizes=[5, 5],
                 out_channels=[10, 20], strides=[1, 1], batch_size=64,
                 preserve_energy=100, flat_size=320, is_debug=False):
        """

        :param input_size:
        :param args: the general arguments for the program, e.g. conv type.
        :param num_classes:
        :param in_channels:
        :param dtype:
        :param kernel_sizes:
        :param out_channels:
        :param strides:
        :param batch_size:
        :param preserve_energy: how much energy to preserve in the input map
        and the filter in the frequency domain.
        :param flat_size: the size of the flat vector after the conv layers.
        """
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.args = args
        self.batch_size = batch_size
        if out_channels is None:
            self.out_channels = [10, 20]
        else:
            self.out_channels = out_channels
        if flat_size is None:
            self.flat_size = 320  # for MNIST dataset
        else:
            self.flat_size = flat_size

        self.is_debug = is_debug

        self.relu = nn.ReLU(inplace=True)
        # For the "same" mode for the convolution, pad the input.
        conv_pads = [0 for _ in kernel_sizes]

        conv = Conv(kernel_sizes=kernel_sizes, in_channels=in_channels,
                    out_channels=out_channels, strides=strides,
                    padding=conv_pads, args=args, is_debug=self.is_debug,
                    dtype=dtype, preserve_energy=preserve_energy)

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1 = conv.get_conv(index=0)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2 = conv.get_conv(index=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(flat_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # it can't be self.batch_size, 320, because:
        # the last batch can be smaller then 64:
        # RuntimeError: shape '[64, 320]' is invalid for input of size 10240,
        # which gives us shape [32, 320].
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)