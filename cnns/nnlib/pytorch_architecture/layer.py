import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Laplace


class Noise(nn.Module):
    def __init__(self, std, noise_form='gauss'):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None
        self.noise_form = noise_form

    def forward(self, x):
        if self.std > 0:
            if self.buffer is None:
                print('noise_form: ', self.noise_form)
                if self.noise_form == 'gauss':
                    self.buffer = torch.zeros_like(
                        x, requires_grad=False).normal_(
                        0, self.std).cuda()
                elif self.noise_form == 'uniform':
                    self.buffer = torch.zeros_like(
                        x, requires_grad=False).uniform_(
                        -self.std, self.std).cuda()
                elif self.noise_form == 'laplace':
                    a = torch.ones_like(x, requires_grad=False).cuda()
                    loc = 0 * a
                    scale = self.std * a
                    m = Laplace(
                        loc=loc,
                        scale=scale,
                    )
                    self.buffer = m.sample()
                else:
                    raise Exception(f'Unknown type of noise (no buffer): {self.noise_form}')
            else:
                if self.noise_form == 'gauss':
                    self.buffer.resize_(x.size()).normal_(0, self.std)
                elif self.noise_form == 'uniform':
                    self.buffer.resize_(x.size()).uniform_(-self.std, self.std)
                elif self.noise_form == 'laplace':
                    a = torch.ones_like(x, requires_grad=False).cuda()
                    loc = 0 * a
                    scale = self.std * a
                    m = Laplace(
                        loc=loc,
                        scale=scale,
                    )
                    self.buffer = m.sample()
                else:
                    raise Exception(f'Unknown type of noise (buffer): {self.noise_form}')

            return x + self.buffer
        return x


class NoiseFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, n):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.
        :param input: the input image
        :param rounder: the rounder object to execute the actual rounding
        """
        # ctx.save_for_backward(input)
        # print("round forward")
        NoiseFunction.mark_dirty(x)
        noise = torch.zeros_like(x).normal_(mean=0, std=n).to(x.device)
        # noise = n.sample(x.shape).squeeze().to(x.device)
        return x + noise

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the
        gradient of the loss with respect to the output, and we need
        to compute the gradient of the loss with respect to the input.
        See: https://arxiv.org/pdf/1706.04701.pdf appendix A
        We do not want to zero out the gradient.
        Defenses that mask a network’s gradients by quantizing
        the input values pose a challenge to gradient-based opti-mization  methods  for  generating  adversarial  examples,such  as  the  procedure  we  describe  in  Section  2.4.   Astraightforward application of the approach would findzero gradients, because small changes to the input do notalter the output at all.  In Section 3.1.1, we describe anapproach where we run the optimizer on a substitute net-work without the color depth reduction step, which ap-proximates the real network.
        """
        # leave the gradient unchanged
        return grad_output.clone(), None


class NoisePassBackward(nn.Module):
    def __init__(self, std):
        super(NoisePassBackward, self).__init__()
        self.n = std
        # self.n = tdist.Normal(torch.tensor([0.0]), torch.tensor([std]))

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.
        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return NoiseFunction.apply(input, self.n)


class BReLU(nn.Module):
    def __init__(self, t=1):
        super(BReLU, self).__init__()
        assert (t > 0)
        self.t = t

    def forward(self, x):
        return x.clamp(0, self.t)


class Normalize(nn.Module):
    def __init__(self, mean_vec, std_vec):
        super(Normalize, self).__init__()
        self.mean = Variable(mean_vec.view(1, 3, 1, 1), requires_grad=False)
        self.std = Variable(std_vec.view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        # x: (batch, 3, H, W)
        # mean, std: (1, 3, 1, 1)
        return (x - self.mean) / self.std
        # return x


class DeNormalize(nn.Module):
    def __init__(self, mean_vec, std_vec):
        super(DeNormalize, self).__init__()
        self.mean = Variable(mean_vec.view(1, 3, 1, 1), requires_grad=False)
        self.std = Variable(std_vec.view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        return x * self.std + self.mean
        # return x
