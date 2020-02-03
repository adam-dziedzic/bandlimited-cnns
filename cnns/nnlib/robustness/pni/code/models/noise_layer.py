import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np

global_noise_type = 'weight'

class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='layerwise',
                 w_noise=True, noise_type=global_noise_type, input_size=None):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param pni:
        :param w_noise:
        :param noise_type: weight or input or both
        """
        super(noise_Linear, self).__init__(in_features, out_features, bias)

        self.pni = pni
        if self.pni is 'layerwise':
            # noise scale for weights
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]),
                                        requires_grad=True)
            # noise scale for inputs
            self.alpha_i = nn.Parameter(torch.Tensor([0.25]),
                                        requires_grad=True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(
                torch.ones(self.out_features).view(-1, 1) * 0.25,
                requires_grad=True)
            self.alpha_i = nn.Parameter(
                torch.ones(self.in_features).view(-1, 1) * 0.25,
                requires_grad=True)

        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size()) * 0.25,
                                        requires_grad=True)
            self.alpha_i = nn.Parameter(torch.ones(input_size) * 0.25,
                                        requires_grad=True)

        self.w_noise = w_noise
        self.noise_type = noise_type

    def forward(self, input):

        if self.noise_type in ('weight', 'both'):
            with torch.no_grad():
                std = self.weight.std().item()
                noise_weight = self.weight.clone().normal_(0, std)

            noise_weight = self.weight + self.alpha_w * noise_weight * self.w_noise
        else:
            noise_weight = self.weight

        if self.noise_type in ('input', 'both'):
            with torch.no_grad():
                std = input.std().item()
                noise_input = input.clone().normal_(0, std)

            noise_input = input + self.alpha_i * noise_input * self.w_noise
        else:
            noise_input = input

        output = F.linear(noise_input, noise_weight, self.bias)

        return output


class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 groups=1, bias=True, pni='layerwise', w_noise=True,
                 noise_type=global_noise_type, input_size=None):
        super(noise_Conv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.pni = pni
        if self.pni is 'layerwise':
            # noise scale for weights
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]),
                                        requires_grad=True)
            # noise scale for inputs
            self.alpha_i = nn.Parameter(torch.Tensor([0.25]),
                                        requires_grad=True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(
                torch.ones(self.out_channels).view(-1, 1, 1, 1) * 0.25,
                requires_grad=True)
            self.alpha_i = nn.Parameter(
                torch.ones(self.in_channels).view(-1, 1, 1, 1) * 0.25,
                requires_grad=True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size()) * 0.25,
                                        requires_grad=True)
            self.alpha_i = nn.Parameter(torch.ones(input_size) * 0.25,
                                        requires_grad=True)

        self.w_noise = w_noise
        self.noise_type = noise_type

    def forward(self, input):
        if self.noise_type in ('weight', 'both'):
            with torch.no_grad():
                std = self.weight.std().item()
                noise_weight = self.weight.clone().normal_(0, std)

            noise_weight = self.weight + self.alpha_w * noise_weight * self.w_noise
        else:
            noise_weight = self.weight

        if self.noise_type in ('input', 'both'):
            with torch.no_grad():
                std = input.std().item()
                noise_input = input.clone().normal_(0, std)

            noise_input = input + self.alpha_i * noise_input * self.w_noise
        else:
            noise_input = input

        output = F.conv2d(noise_input, noise_weight, self.bias, self.stride,
                          self.padding, self.dilation,
                          self.groups)

        return output
