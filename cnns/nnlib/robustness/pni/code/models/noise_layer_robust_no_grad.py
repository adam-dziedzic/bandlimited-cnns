import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
import os
import torch


class noise_Conv2dFunction3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, noise_std, noise_type):
        # torch.autograd.set_detect_anomaly(True)
        input = input.clone()
        ctx.save_for_backward(input)
        if noise_std > 0:
            if noise_type == 'gauss':
                noise_i = input.clone().detach().normal_(
                    0, noise_std)
                # noise_i = torch.randn(input.size()) * noise_std
                # noise_i = noise_i.to(input.device).to(input.dtype)
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

        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class noise_Conv2dFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                input,
                noise_std,
                noise_type,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups
                ):
        # torch.autograd.set_detect_anomaly(True)
        input = input.clone()
        ctx.save_for_backward(input)
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

        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None, None


def replace_fun(x):
    """
    Replacement function for backward propagation.
    By default, it is an identity function.

    :param x: the input in the forward pass
    :return: the output for the replaced forward layer with this function

    """
    return x


class noise_Conv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        # torch.autograd.set_detect_anomaly(True)
        kwargs = args[0]
        input = kwargs['input']
        input = input.clone()
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
        cur_dir = os.getcwd()
        print('cur_dir: ', cur_dir)

        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        cur_dir = os.getcwd()
        with open(cur_dir + '/grads.txt', 'a') as f:
            f.write(str(grad_outputs) + '\n')
        # saved_tensors = ctx.saved_tensors
        # input = saved_tensors[0]
        # input = input.detach().clone().requires_grad_()
        # # replace the noise addition with an identity function
        # output = replace_fun(x=input)
        # grad = torch.autograd.grad(
        #     outputs=output, inputs=input, grad_outputs=grad_outputs)
        return 'adam'


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

        noise_type = self.noise_type
        noise_std = self.noise_std
        weight = self.weight
        bias = self.bias
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        # return noise_Conv2dFunction.apply(kwargs)

        # return noise_Conv2dFunction2.apply(
        #     input,
        #     noise_std,
        #     noise_type,
        #     weight,
        #     bias,
        #     stride,
        #     padding,
        #     dilation,
        #     groups
        # )

        noisy_input = noise_Conv2dFunction3.apply(
            input, noise_std, noise_type,
        )

        output = F.conv2d(
            noisy_input, weight=weight, bias=bias, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups)

        return output
