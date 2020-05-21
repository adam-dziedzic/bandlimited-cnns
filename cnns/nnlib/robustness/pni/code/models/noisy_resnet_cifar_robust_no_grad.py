import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .noise_layer_robust_no_grad import noise_Conv2d


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 inner_noise=0.1, noise_type='gauss'):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = noise_Conv2d(inplanes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False,
                                   noise_std=inner_noise,
                                   noise_type=noise_type)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = noise_Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=1, bias=False,
                                   noise_std=inner_noise,
                                   noise_type=noise_type)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=False)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=False)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, init_noise,
                 inner_noise, noise_type):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
          init_noise: init noise std
          inner_noise: inner noise std
          noise_type: type of noise (e.g. gauss or uniform)
        """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (
                       depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(
            depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = noise_Conv2d(3, 16, kernel_size=3, stride=1,
                                       padding=1, bias=False,
                                       noise_std=init_noise,
                                       noise_type=noise_type)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block=block, planes=16, blocks=layer_blocks, stride=1, inner_noise=inner_noise,
                                        noise_type=noise_type)
        self.stage_2 = self._make_layer(block=block, planes=32, blocks=layer_blocks, stride=2, inner_noise=inner_noise,
                                        noise_type=noise_type)
        self.stage_3 = self._make_layer(block=block, planes=64, blocks=layer_blocks, stride=2, inner_noise=inner_noise,
                                        noise_type=noise_type)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, inner_noise=0.1,
                    noise_type='gauss'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion,
                                     stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes,
                                planes=planes,
                                inner_noise=inner_noise,
                                noise_type=noise_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=False)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def noise_resnet20_robust_no_grad(
        num_classes=10, init_noise=0.2, inner_noise=0.1, noise_type='gauss'):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(
        block=ResNetBasicblock,
        depth=20,
        num_classes=num_classes,
        init_noise=init_noise,
        inner_noise=inner_noise,
        noise_type=noise_type,
    )
    return model


