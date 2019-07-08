import torch
from torch.nn import Module
from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack
from foolbox.attacks.additive_noise import AdditiveGaussianNoiseAttack
from cnns.nnlib.robustness.utils import AdditiveLaplaceNoiseAttack
import numpy as np


class NoiseFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, noiser, noise_level, min_, max_):
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
        NoiseFunction.mark_dirty(input)
        N, C, H, W = input.shape
        dtype = np.float
        if input.dtype == torch.double:
            dtype = np.double
        dummy_image_for_type = np.zeros((N, C, H, W), dtype=dtype)
        noise = noiser._sample_noise(
            epsilon=noise_level,
            bounds=(min_, max_), image=dummy_image_for_type)
        noise = torch.from_numpy(noise).to(input.dtype).to(input.device)
        return input + noise

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the
        gradient of the loss with respect to the output, and we need
        to compute the gradient of the loss with respect to the input.

        See: https://arxiv.org/pdf/1706.04701.pdf appendix A

        We do not want to zero out the gradient.

        Defenses that mask a network’s gradients by quantizingthe input values pose a challenge to gradient-based opti-mization  methods  for  generating  adversarial  examples,such  as  the  procedure  we  describe  in  Section  2.4.   Astraightforward application of the approach would findzero gradients, because small changes to the input do notalter the output at all.  In Section 3.1.1, we describe anapproach where we run the optimizer on a substitute net-work without the color depth reduction step, which ap-proximates the real network.
        """
        # print("round backward")
        return grad_output.clone(), None, None, None, None


class Noise(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """

    def __init__(self, args):
        super(Noise, self).__init__()
        if args.noise_sigma > 0:
            # gauss_image = gauss(image_numpy=image, sigma=args.noise_sigma)
            self.noiser = AdditiveGaussianNoiseAttack()
            self.noise_level = args.noise_sigma
        elif args.noise_epsilon > 0:
            self.noiser = AdditiveUniformNoiseAttack()
            self.noise_level = args.noise_epsilon
        elif args.laplace_epsilon > 0:
            self.noiser = AdditiveLaplaceNoiseAttack(args=args)
            self.noise_level = args.laplace_epsilon
        self.min = args.min
        self.max = args.max

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return NoiseFunction.apply(input, self.noiser, self.noise_level,
                                   self.min, self.max)


class NoiseGauss(Noise):

    def __init__(self, args):
        super(NoiseGauss, self).__init__(args=args)
        # overwrite the noiser
        if args.noise_sigma > 0:
            self.noiser = AdditiveGaussianNoiseAttack()
            self.noise_level = args.noise_sigma


class NoiseUniform(Noise):

    def __init__(self, args):
        super(NoiseUniform, self).__init__(args=args)
        if args.noise_epsilon > 0:
            self.noiser = AdditiveUniformNoiseAttack()
            self.noise_level = args.noise_epsilon


class NoiseLaplace(Noise):

    def __init__(self, args):
        super(NoiseLaplace, self).__init__(args=args)
        if args.laplace_epsilon > 0:
            self.noiser = AdditiveLaplaceNoiseAttack(args=args)
            self.noise_level = args.laplace_epsilon
