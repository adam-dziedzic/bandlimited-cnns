import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from cnns.nnlib.pytorch_layers.conv_picker import Conv

class FCNNPytorch(nn.Module):

    # @profile
    def __init__(self, args, kernel_sizes=[8, 5, 3],
                 out_channels=[128, 256, 128], strides=[1, 1, 1]):
        """
        Create the FCNN model in PyTorch.

        :param args: the general arguments (conv type, debug mode, etc).
        :param dtype: global - the type of pytorch data/weights.
        :param kernel_sizes: the sizes of the kernels in each conv layer.
        :param out_channels: the number of filters for each conv layer.
        :param strides: the strides for the convolutions.
        """
        super(FCNNPytorch, self).__init__()
        # input_size: the length (width) of the time series.
        self.input_size = args.input_size
        # num_classes: number of output classes.
        self.num_classes = args.num_classes
        # in_channels: number of channels in the input data.
        self.in_channels = args.in_channels
        self.dtype = args.dtype
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.strides = strides
        self.conv_type = args.conv_type
        self.is_debug = args.is_debug
        self.preserve_energy = args.preserve_energy

        self.relu = nn.ReLU(inplace=True)
        # For the "same" mode for the convolution, pad the input.
        conv_pads = [kernel_size - 1 for kernel_size in kernel_sizes]

        conv = Conv(kernel_sizes=kernel_sizes, in_channels=self.in_channels,
                    out_channels=out_channels, strides=strides,
                    padding=conv_pads, args=args)

        index = 0
        self.conv0 = conv.get_conv(index=index)
        self.bn0 = nn.BatchNorm1d(num_features=out_channels[index])

        index = 1
        self.conv1 = conv.get_conv(index=index)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels[index])

        index = 2
        self.conv2 = conv.get_conv(index=index)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels[index])
        self.lin = nn.Linear(out_channels[index], self.num_classes)

    def pad_out(self, out, index):
        """
        Pad the output to keep the size of the processed input the same through
        all the layers.

        :param out: the output of the previous layer
        :param index: index of the conv layer.
        :return:
        """
        if self.kernel_sizes[index] % 2 == 0:
            # If kernel size is even, add one more padding value on the right.
            out = F.pad(out, (0, 1), "constant", 0)
        return out

    def forward(self, out):
        """
        The forward pass through the network.

        :param out: the input data for the network.
        :return: the output class.
        """

        # 0th layer.
        index = 0
        out = self.pad_out(out, index)
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu(out)

        # 1st layer.
        index = 1
        out = self.pad_out(out, index)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 2nd layer.
        index = 2
        out = self.pad_out(out, index)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Classification.
        # Average across the channels.
        # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
        # In Keras it is implemented as: K.mean(inputs, axis=1). The channel is
        # the last dimension in Keras.
        out = torch.mean(out, dim=2)
        out = self.lin(out)

        # To imitate the cross entropy loss with the nll (negative log
        # likelihood) loss.
        out = log_softmax(out, dim=-1)

        return out