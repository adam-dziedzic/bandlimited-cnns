import torch.nn as nn
import torch.nn.functional as F


class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 groups=1, bias=True, noise_std=0.1):
        super(noise_Conv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding, dilation, groups, bias)
        self.noise_std = noise_std

    def forward(self, input):
        noise_i = input.clone().normal_(0, self.noise_std)
        noise_input = input + noise_i

        output = F.conv2d(noise_input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation,
                          self.groups)

        return output
