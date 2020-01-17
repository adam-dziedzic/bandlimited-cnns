'''VGG11/13/16/19 in Pytorch.

We perturb the weight and bias parameters for convolutional, linear and batch
normalization layers.

'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from cnns.nnlib.pytorch_architecture import layer

Noise = layer.Noise
NoisePassBackward = layer.NoisePassBackward

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}

fc_noise = [0.02517237886786461, 0.019698506221175194]
conv_bn_noise = [0.10803990811109543, 0.11500568687915802, 0.0, 0.0,
                 0.0240741278976202, 0.02496134676039219, 0.0, 0.0,
                 0.024029752239584923, 0.022710483521223068, 0.0, 0.0,
                 0.017008690163493156, 0.018482772633433342, 0.0, 0.0,
                 0.017002755776047707, 0.017049072310328484, 0.0, 0.0,
                 0.012026282958686352, 0.011528636328876019, 0.0, 0.0,
                 0.012020026333630085, 0.011991619132459164, 0.0, 0.0,
                 0.01202633511275053, 0.011740812100470066, 0.0, 0.0,
                 0.008503676392138004, 0.00861838273704052, 0.0, 0.0,
                 0.008504466153681278, 0.0082998713478446, 0.0, 0.0,
                 0.008508573286235332, 0.008545858785510063, 0.0, 0.0,
                 0.008508553728461266, 0.008277240209281445, 0.0, 0.0,
                 0.008508582599461079, 0.008451293222606182, 0.0, 0.0]

x_noise = [
    0.250856459,
    0.243855923,
    0.25971508,
    0.130152896,
    0.17995508,
    0.09655977,
    0.078154489,
    0.120438933,
    0.073042072,
    0.047605705,
    0.061036598,
    0.04321336,
    0.050626367,
]


def perturb_param(param, param_noise, buffer_noise):
    if param_noise > 0:
        buffer_noise.normal_(0, param_noise)
    return param + buffer_noise


class Conv2dNoise(nn.Conv2d):
    """
    For the conv layer we perturb the convolutional filters and bias terms.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 weight_noise=0.0, bias_noise=0.0):
        super(Conv2dNoise, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          padding=padding)
        self.weight_noise = weight_noise
        self.bias_noise = bias_noise
        self.buffer_weight_noise = None
        self.buffer_bias_noise = None

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.buffer_weight_noise is None:
            self.buffer_weight_noise = torch.zeros_like(
                self.weight, requires_grad=False)
            if self.weight_noise > 0:
                self.buffer_weight_noise.normal_(
                    0, self.weight_noise).to(self.weight.device)
        weight = perturb_param(param=self.weight,
                               param_noise=self.weight_noise,
                               buffer_noise=self.buffer_weight_noise)
        if self.buffer_bias_noise is None:
            self.buffer_bias_noise = torch.zeros_like(
                self.bias, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_bias_noise.normal_(
                    0, self.param_noise).to(self.bias.device)
        bias = perturb_param(param=self.bias,
                             param_noise=self.bias_noise,
                             buffer_noise=self.buffer_bias_noise)
        return self.conv2d_forward(input, weight, bias)


class LinearNoise(nn.Linear):
    """
    For the linear layer we perturb the weights and the additive bias.
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_noise=0.0, bias_noise=0.0):
        super(LinearNoise, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        self.weight_noise = weight_noise
        self.bias_noise = bias_noise
        self.buffer_weight_noise = None
        self.buffer_bias_noise = None

    def forward(self, input):
        if self.buffer_weight_noise is None:
            self.buffer_weight_noise = torch.zeros_like(
                self.weight, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_weight_noise.normal_(
                    0, self.param_noise).to(self.weight.device)
        weight = perturb_param(param=self.weight,
                               param_noise=self.weight_noise,
                               buffer_noise=self.buffer_weight_noise)
        if self.buffer_bias_noise is None:
            self.buffer_bias_noise = torch.zeros_like(
                self.bias, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_bias_noise.normal_(
                    0, self.param_noise).to(self.bias.device)
        bias = perturb_param(param=self.bias,
                             param_noise=self.bias_noise,
                             buffer_noise=self.buffer_bias_noise)
        return F.linear(input, weight, bias)


class BatchNorm2dNoise(nn.BatchNorm2d):
    def __init__(self, num_features,
                 weight_noise=0.0, bias_noise=0.0):
        super(BatchNorm2dNoise, self).__init__(num_features=num_features)
        self.weight_noise = weight_noise
        self.bias_noise = bias_noise
        self.buffer_weight_noise = None
        self.buffer_bias_noise = None

    def forward(self, input):
        self._check_input_dim(input)

        if self.buffer_weight_noise is None:
            self.buffer_weight_noise = torch.zeros_like(
                self.weight, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_weight_noise.normal_(
                    0, self.param_noise).to(self.weight.device)
        weight = perturb_param(param=self.weight,
                               param_noise=self.weight_noise,
                               buffer_noise=self.buffer_weight_noise)
        if self.buffer_bias_noise is None:
            self.buffer_bias_noise = torch.zeros_like(
                self.bias, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_bias_noise.normal_(
                    0, self.param_noise).to(self.bias.device)
        bias = perturb_param(param=self.bias,
                             param_noise=self.bias_noise,
                             buffer_noise=self.buffer_bias_noise)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class VGG(nn.Module):
    def __init__(self, vgg_name, param_noise=-0.1, noise_type='standard'):
        super(VGG, self).__init__()
        self.param_noise = param_noise
        self.noise_type = noise_type
        self.features = self._make_layers(cfg[vgg_name])
        fc_weight_noise = self.adjust_data_param_noise(fc_noise[0])
        fc_bias_noise = self.adjust_data_param_noise(fc_noise[1])
        self.classifier = LinearNoise(512, 10,
                                      weight_noise=fc_weight_noise,
                                      bias_noise=fc_bias_noise)

    def adjust_data_param_noise(self, x):
        """
        set param noise
        :param x: input param or data point
        :return: adjusted param or data point
        """
        return x + self.param_noise * x

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        conv_bn_index = 0
        noise_x_index = 0
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                noise = x_noise[noise_x_index]
                noise = self.adjust_data_param_noise(noise)
                noise_x_index += 1
                if self.noise_type == 'backward':
                    noise_layer = NoisePassBackward(noise)
                elif self.noise_type == 'standard':
                    noise_layer = Noise(noise)
                else:
                    raise Exception(f'Unknown noise type: {self.noise_type}')

                print('conv_bn_index: ', conv_bn_index)
                # specify noise for each layer and its weight and bias params
                conv_weight_noise = conv_bn_noise[conv_bn_index + 0]
                conv_bias_noise = conv_bn_noise[conv_bn_index + 1]
                bn_weight_noise = conv_bn_noise[conv_bn_index + 2]
                bn_bias_noise = conv_bn_noise[conv_bn_index + 3]
                conv_bn_index += 4

                conv_weight_noise = self.adjust_data_param_noise(
                    conv_weight_noise)
                conv_bias_noise = self.adjust_data_param_noise(conv_bias_noise)
                bn_weight_noise = self.adjust_data_param_noise(bn_weight_noise)
                bn_bias_noise = self.adjust_data_param_noise(bn_bias_noise)

                layers += [
                    noise_layer,
                    Conv2dNoise(in_channels, x, kernel_size=3, padding=1,
                                weight_noise=conv_weight_noise,
                                bias_noise=conv_bias_noise),
                    BatchNorm2dNoise(x, weight_noise=bn_weight_noise,
                                     bias_noise=bn_bias_noise),
                    nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
