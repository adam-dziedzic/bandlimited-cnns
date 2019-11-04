import torch.nn as nn
import torch.nn.functional as F
from cnns.nnlib.pytorch_layers.conv_picker import Conv
from cnns.nnlib.pytorch_layers.conv1D_fft import Conv1dfft
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.robustness.channels.channels_definition import get_svd_index
from cnns.nnlib.pytorch_architecture.net import conv_param_nr


def get_conv(args, in_channels, out_channels, kernel_size, stride=1,
             padding=0, bias=True):
    return Conv(kernel_sizes=[kernel_size], in_channels=in_channels,
                out_channels=[out_channels], strides=[stride],
                padding=[padding], args=args, is_bias=bias).get_conv()


H = 28
W = 28
num_classes = 10

kernel_size1 = 5
kernel_size2 = 5
hidden_neurons = 500
pull1 = 2
pull2 = 2
# out_channels2 = 10
out_channels2 = 50


def get_HW_after_pull2(
        H=H,
        W=W,
        kernel_size1=kernel_size1,
        kernel_size2=kernel_size2,
        pull1=pull1,
        pull2=pull2):
    H_conv1 = H - (kernel_size1 - 1)
    W_conv1 = W - (kernel_size1 - 1)
    assert H_conv1 % 2 == 0
    assert W_conv1 % 2 == 0
    H_pull1 = H_conv1 // pull1  # 12
    W_pull1 = W_conv1 // pull1  # 12
    W_conv2 = W_pull1 - (kernel_size2 - 1)  # 8
    H_conv2 = H_pull1 - (kernel_size2 - 1)  # 8
    assert H_conv2 % pull2 == 0
    assert W_conv2 % pull2 == 0
    H_pull2 = H_conv2 // pull2  # 4
    W_pull2 = W_conv2 // pull2  # 4
    return H_pull2, W_pull2


class NetSyntheticSVD(nn.Module):
    def __init__(self, args):
        super(NetSyntheticSVD, self).__init__()
        self.args = args

        conv_type_2D = args.conv_type
        conv_type_1D = ConvType.STANDARD
        H = args.input_height
        W = args.input_width
        compress_rate = args.svd_transform
        index = get_svd_index(H=H, W=W, compress_rate=compress_rate)
        print('svd index in NetSynthetic SVD: ', index)
        in_channels = index

        # self.in_channels2 = out_channels1
        in_channels2 = 1  # fixed by SVD
        conv2_param_nr = in_channels2 * out_channels2 * kernel_size2  # 250
        conv1_param_nr = conv_param_nr - conv2_param_nr  # 5100 - 250 = 4850
        # conv1_param_nr = in_channels1 (i) * self.out_channels1 * kernel_size1
        # for in_channels = 13, out_channels1 = 4850 / (13 * 5) = 3
        out_channels1 = 1 * int(conv1_param_nr  / (in_channels * kernel_size1))
        # out_channels1 = 5
        print('out channels in NetSythetic SVD: ', index)

        args.conv_type = conv_type_1D
        self.conv1_u = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=kernel_size1,
                                stride=1)

        self.conv1_s = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=1,
                                stride=1)

        self.conv1_v = get_conv(args, in_channels=in_channels,
                                out_channels=out_channels1,
                                kernel_size=kernel_size1,
                                stride=1)



        args.conv_type = conv_type_2D
        self.conv2 = get_conv(args, in_channels=in_channels2,
                              out_channels=out_channels2,
                              kernel_size=kernel_size2, stride=1)

        H_after_pull2, W_after_pull2 = get_HW_after_pull2()
        self.fc1 = nn.Linear(
            H_after_pull2 * W_after_pull2 * out_channels2,
            hidden_neurons)
        self.fc2 = nn.Linear(
            hidden_neurons,
            args.num_classes)

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


        u = F.max_pool1d(u, pull1)
        v = F.max_pool1d(v, pull1)

        # Combine the singular vectors and singular values to the 2D
        # representation.
        u = u.transpose(2, 1)
        s = s.transpose(2, 1)
        u_s = u * s
        x = u_s.bmm(v)
        # Add a single channel.
        x = x.unsqueeze(1)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, pull2, pull2)
        H_after_pull2, W_after_pull2 = get_HW_after_pull2()
        x = x.view(-1,
                   H_after_pull2 * W_after_pull2 * out_channels2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
