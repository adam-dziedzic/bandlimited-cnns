'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
from cnns.nnlib.pytorch_layers.conv_picker import Conv
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import CompressType
import torch

args = Arguments()
args.conv_type = ConvType.FFT2D
args.compress_rate = 80.0
args.dtype = torch.float
args.preserve_energy = None
args.next_power2 = True
args.is_debug = False
args.compress_type = CompressType.STANDARD


def conv3x3(in_planes, out_planes, compress_rate = args.compress_rate,
            stride=1, padding=1, args=args):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                      padding=1, bias=False)
    args.compress_rate = compress_rate
    return Conv(kernel_sizes=[3], in_channels=in_planes,
                out_channels=[out_planes], strides=[stride],
                padding=[padding], args=args, is_bias=False).get_conv()


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, compress_rate=args.compress_rate):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name],
                                          compress_rate=compress_rate)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, compress_rate=args.compress_rate):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes=in_channels,
                                   out_planes=x,
                                   padding=1,
                                   compress_rate=compress_rate),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

