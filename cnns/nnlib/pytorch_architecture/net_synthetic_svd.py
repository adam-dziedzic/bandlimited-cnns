import torch.nn as nn
import torch.nn.functional as F
from cnns.nnlib.pytorch_layers.conv_picker import Conv
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfft
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.robustness.channels.channels_definition import get_svd_index


def get_conv(args, in_channels, out_channels, kernel_size, stride=1,
             padding=0, bias=True):
    return Conv(kernel_sizes=[kernel_size], in_channels=in_channels,
                out_channels=[out_channels], strides=[stride],
                padding=[padding], args=args, is_bias=bias).get_conv()


class NetSyntheticSVD(nn.Module):
    def __init__(self, args):
        super(NetSyntheticSVD, self).__init__()
        self.args = args
        out_channels1 = 5
        conv_type_2D = args.conv_type
        conv_type_1D = ConvType.STANDARD
        H = args.input_height
        W = args.input_width
        compress_rate = args.svd_transform
        index = get_svd_index(H=H, W=W, compress_rate=compress_rate)
        in_channels = index
        first_kernel_size = 5
        args.conv_type = conv_type_1D
        self.conv1_u = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=first_kernel_size,
                                stride=1)

        self.conv1_s = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=1,
                                stride=1)

        self.conv1_v = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=first_kernel_size,
                                stride=1)

        self.out_channels2 = 10
        # self.in_channels2 = out_channels1
        self.in_channels2 = 1
        args.conv_type = conv_type_2D
        self.conv2 = get_conv(args, in_channels=self.in_channels2,
                              out_channels=self.out_channels2,
                              kernel_size=5, stride=1)

        self.fc1 = nn.Linear(4 * 4 * self.out_channels2, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, data):
        u = data['u']
        s = data['s']
        v = data['v']

        u = self.conv1_u(u)
        s = self.conv1_s(s)
        v = self.conv1_v(v)

        u = F.relu(u)
        s = F.relu(s)
        v = F.relu(v)

        u = F.max_pool1d(u, 2)
        v = F.max_pool1d(v, 2)

        # Combine the singular vectors and singular values to the 2D
        # representation.
        u = u.transpose(2, 1)
        s = s.transpose(2, 1)
        u_s = u * s
        x = u_s.matmul(v)
        # Add a single channel.
        x = x.unsqueeze(1)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * self.out_channels2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
