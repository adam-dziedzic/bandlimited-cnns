import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
import torch


class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noise_std=0.1,
                 noise_type='gauss'):
        super(noise_Conv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding, dilation, groups, bias)
        self.noise_std = noise_std
        self.noise_type = noise_type

    def forward(self, input):
        # print('noise_std: ', self.noise_std)
        # print('noise type: ', self.noise_type)
        if self.noise_type == 'gauss':
            noise_i = input.clone().normal_(0, self.noise_std)
        elif self.noise_type == 'uniform':
            noise_i = input.clone().uniform_(
                -self.noise_std, self.noise_std)
        elif self.noise_type == 'laplace':
            a = torch.ones_like(input)
            loc = 0 * a
            scale = self.noise_std * a
            m = Laplace(
                loc=loc,
                scale=scale,
            )
            noise_i = m.sample()
        else:
            raise Exception(f'Unknown noise type: {self.noise_type}')
        noise_input = input + noise_i

        output = F.conv2d(noise_input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation,
                          self.groups)

        return output
