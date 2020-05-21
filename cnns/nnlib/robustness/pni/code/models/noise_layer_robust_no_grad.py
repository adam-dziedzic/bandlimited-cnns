import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
from torch.distributions import Normal
import torch


class noise_Conv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        # torch.autograd.set_detect_anomaly(True)
        kwargs = args[0]
        input = kwargs['input']
        ctx.save_for_backward(input)
        # print('kwargs: ', kwargs.keys())
        noise_std = kwargs['noise_std']
        noise_type = kwargs['noise_type']
        if noise_std > 0:
            if noise_type == 'gauss':
                # noise_i = input.clone().detach().normal_(
                #     0, noise_std)
                noise_i = torch.randn(input.size()) * noise_std
                noise_i = noise_i.to(input.device).to(input.dtype)
            elif noise_type == 'uniform':
                noise_i = input.clone().detach().uniform(
                    -noise_std, noise_std)
            elif noise_type == 'laplace':
                a = torch.ones_like(input)
                loc = 0 * a
                scale = noise_std * a
                m = Laplace(
                    loc=loc,
                    scale=scale,
                )
                noise_i = m.sample()
            else:
                raise Exception(f'Unknown noise type: {noise_type}')
            input += noise_i

        weight = kwargs['weight']
        bias = kwargs['bias']
        stride = kwargs['stride']
        padding = kwargs['padding']
        dilation = kwargs['dilation']
        groups = kwargs['groups']

        output = F.conv2d(
            input, weight=weight, bias=bias, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups)

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved_tensors = ctx.saved_tensors
        input = saved_tensors[0]
        input = input.detach().clone().requires_grad_()
        # replace the noise addition with an identity function
        grad = torch.autograd.grad(
            outputs=input, inputs=input, grad_outputs=grad_outputs)
        return grad


class noise_Conv2d(nn.Conv2d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=True, noise_std=0.1,
            noise_type='gauss'):
        super(noise_Conv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride,
            padding, dilation, groups, bias)
        self.noise_std = noise_std
        self.noise_type = noise_type

    def forward(self, input):
        kwargs = {
            'input': input,
            'noise_type': self.noise_type,
            'noise_std': self.noise_std,
            'weight': self.weight,
            'bias': self.bias,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
        }
        return noise_Conv2dFunction.apply(kwargs)
