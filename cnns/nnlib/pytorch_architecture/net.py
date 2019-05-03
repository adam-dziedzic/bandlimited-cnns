import torch.nn as nn
import torch.nn.functional as F
from cnns.nnlib.pytorch_layers.conv_picker import Conv


def get_conv(args, in_channels, out_channels, kernel_size, stride=1,
             padding=0, bias=True):
    return Conv(kernel_sizes=[kernel_size], in_channels=in_channels,
                out_channels=[out_channels], strides=[stride],
                padding=[padding], args=args, is_bias=bias).get_conv()


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1 = get_conv(args, in_channels=1, out_channels=20,
                              kernel_size=5, stride=1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2 = get_conv(args, in_channels=20, out_channels=50,
                              kernel_size=5, stride=1)

        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
