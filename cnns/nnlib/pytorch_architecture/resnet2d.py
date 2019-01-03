import time
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from cnns.nnlib.pytorch_layers.conv_picker import Conv
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfft
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.arguments import Arguments

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, args=None):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                      padding=1, bias=False)
    return Conv(kernel_sizes=[3], in_channels=in_planes,
                out_channels=[out_planes], strides=[stride],
                padding=[1], args=args, is_bias=False).get_conv()


def conv1x1(in_planes, out_planes, stride=1, args=None):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)
    # It is rather unnecessary to use fft convolution for kernels of size 1x1.
    # return Conv(kernel_sizes=[1], in_channels=in_planes,
    #             out_channels=[out_planes], strides=[stride],
    #             padding=[0], args=args, is_bias=False).get_conv()

# global_block_conv1_time = 0.0
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, args=None):
        super(BasicBlock, self).__init__()
        self.args = args
        self.conv1 = conv3x3(inplanes, planes, stride, args=args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, args=args)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # start_conv1 = time.time()

        out = self.conv1(x)

        # global global_block_conv1_time
        # global_block_conv1_time += time.time() - start_conv1
        # print("global_block_conv1_time: ", global_block_conv1_time)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, args=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, args=args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, args=args)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, args=args)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, args=None):
        super(ResNet, self).__init__()
        # self.global_layer1_time = 0.0
        self.inplanes = 64
        self.conv1 = Conv(kernel_sizes=[3], in_channels=args.in_channels,
                          out_channels=[64], strides=[1], padding=[1],
                          args=args, is_bias=False).get_conv()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], args=args)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       args=args)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       args=args)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       args=args)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, args.num_classes)
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dfft):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, args=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        args=args),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            args=args))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # start_time = time.time()
        x = self.layer1(x)
        # self.global_layer1_time += time.time() - start_time
        # print("global layer1 time: ", self.global_layer1_time)
        # print("x after layer 1: ", x[0])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class Args(object):
    pass


if __name__ == "__main__":
    dtype = torch.float
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device used: ", str(device))

    args = Arguments()
    args.in_channels = 3
    # args.conv_type = "FFT2D"
    args.conv_type = ConvType.STANDARD2D
    args.index_back = None
    args.preserve_energy = None
    args.is_debug = False
    args.next_power2 = True
    args.compress_type = CompressType.STANDARD
    args.tensor_type = TensorType.FLOAT32
    args.num_classes = 10
    args.min_batch_size = 16
    args.test_batch_size = 16

    batch_size = 16
    inputs = torch.randn(batch_size, args.in_channels, 32, 32, dtype=dtype,
                         device=device)

    model = resnet18(args=args)
    model.to(device)
    model.eval()
    start_eval = time.time()
    outputs_standard = model(inputs)
    standard_time = time.time() - start_eval
    print("standard eval time: ", standard_time)

    print("outputs standard: ", outputs_standard)

    args.conv_type = ConvType.FFT2D

    model = resnet18(args=args)
    model.to(device)
    model.eval()
    start_eval = time.time()
    outputs_fft = model(inputs)
    fft_time = time.time() - start_eval
    print("conv2D FFT time: ", fft_time)

    print("outputs fft: ", outputs_fft)

    print("pytorch speedup over fft for testing resnet18: ",
          fft_time / standard_time)
