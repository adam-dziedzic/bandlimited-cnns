import torch.nn as nn
import torch.nn.functional as F
import torch

global_noise_type = 'both'


class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 noise_type=global_noise_type):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param pni:
        :param w_noise:
        :param noise_type: weight or input or both
        """
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        self.noise_type = noise_type

    def forward(self, input):
        if self.noise_type in ('weight', 'both'):
            with torch.no_grad():
                std = self.weight.std().item()
                noise_w = self.weight.clone().normal_(0, std)

            noise_weight = self.weight + noise_w
        else:
            noise_weight = self.weight

        if self.noise_type in ('input', 'both'):
            with torch.no_grad():
                std = input.std().item()
                noise_i = input.clone().normal_(0, std)

            noise_input = input + noise_i
        else:
            noise_input = input

        output = F.linear(noise_input, noise_weight, self.bias)

        return output


class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 groups=1, bias=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding, dilation, groups, bias)

    def forward(self, input):
        if self.noise_type in ('weight', 'both'):
            with torch.no_grad():
                std = self.weight.std().item()
                noise_w = self.weight.clone().normal_(0, std)

            noise_weight = self.weight + noise_w
        else:
            noise_weight = self.weight

        if self.noise_type in ('input', 'both'):
            with torch.no_grad():
                std = input.std().item()
                noise_i = input.clone().normal_(0, std)
            noise_input = input + noise_i
        else:
            noise_input = input

        output = F.conv2d(noise_input, noise_weight, self.bias, self.stride,
                          self.padding, self.dilation,
                          self.groups)

        return output
