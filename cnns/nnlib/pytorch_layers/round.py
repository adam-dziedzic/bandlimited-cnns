import torch
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from torch.nn import Module


class RoundFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, rounder):
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
        RoundFunction.mark_dirty(input)
        return rounder(input)

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
        return grad_output.clone(), None


class Round(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """
    """
    Rounding layer.
    """

    def __init__(self, args):
        super(Round, self).__init__()
        self.rounder = DenormRoundNorm(
            values_per_channel=args.values_per_channel,
            mean_array=args.mean_array,
            std_array=args.std_array, device=args.device)

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return RoundFunction.apply(input, self.rounder)
