'''VGG11/13/16/19 in Pytorch.

We perturb the weight and bias parameters for convolutional, linear and batch
normalization layers.

'''
import torch.nn as nn
import torch
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def perturb_param(param, param_noise, buffer_noise):
    if param_noise > 0:
        buffer_noise.normal_(0, param_noise)
    return param + buffer_noise


class BatchNorm2dNoise(nn.BatchNorm2d):
    def __init__(self, num_features,
                 param_noise=0.04):
        super(BatchNorm2dNoise, self).__init__(num_features=num_features)
        self.param_noise = param_noise
        self.buffer_weight_noise = None

    def forward(self, input):
        self._check_input_dim(input)

        if self.buffer_weight_noise is None:
            self.buffer_weight_noise = torch.zeros_like(
                self.weight, requires_grad=False)
            if self.param_noise > 0:
                self.buffer_weight_noise.normal_(
                    0, self.param_noise).to(self.weight.device)
        weight = perturb_param(param=self.weight,
                               param_noise=self.param_noise,
                               buffer_noise=self.buffer_weight_noise)
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
            input, self.running_mean, self.running_var, weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class VGG(nn.Module):
    def __init__(self, vgg_name, param_noise=0.04):
        super(VGG, self).__init__()
        self.param_noise = param_noise
        self.classifier = nn.Linear(512, 10)
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    BatchNorm2dNoise(x, param_noise=self.param_noise),
                    nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
