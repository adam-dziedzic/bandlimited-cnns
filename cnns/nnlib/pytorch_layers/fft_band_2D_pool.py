import torch
from torch.nn import Module
from cnns.nnlib.utils.general_utils import next_power2
from torch.nn.functional import pad as torch_pad
import numpy as np
from cnns.nnlib.utils.shift_DC_component import shift_DC
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_index_forward
from cnns.nnlib.pytorch_layers.pytorch_utils import \
    compress_2D_index_forward_full


class FFTBandFunction2DPool(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    signal_ndim = 2

    @staticmethod
    def forward(ctx, input, args, onesided=True, is_test=False):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.

        :param input: the input image
        :param args: for compress rate and next_power2.
        :param onesided: FFT convolution leverages the conjugate symmetry and
        returns only roughly half of the FFT map, otherwise the full map is
        returned
        :param is_test: test if the number of zero-ed out coefficients is
        correct
        """
        # ctx.save_for_backward(input)
        # print("round forward")
        FFTBandFunction2DPool.mark_dirty(input)

        N, C, H, W = input.size()

        if H != W:
            raise Exception(f"We support only squared input but the width: {W}"
                            f" is differnt from height: {H}")

        if args.next_power2:
            H_fft = next_power2(H)
            W_fft = next_power2(W)
            pad_H = H_fft - H
            pad_W = W_fft - W
            input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)

        xfft = torch.rfft(input, signal_ndim=FFTBandFunction2DPool.signal_ndim,
                          onesided=onesided)

        del input
        _, _, H_xfft, W_xfft, _ = xfft.size()

        # r - is the side of the retained square in one of the quadrants
        # 4 * r ** 2 / (H * W) = (1 - c)
        # r = np.sqrt((1 - c) * (H * W) / 4)

        compress_rate = args.compress_rate / 100

        if onesided:
            divisor = 2
        else:
            divisor = 4

        # r - is the length of the side that we retain after compression.
        r = np.sqrt((1 - compress_rate) * H_xfft * W_xfft / divisor)
        # r = np.floor(r)
        r = np.ceil(r)
        r = int(r)

        if onesided:
            xfft = compress_2D_index_forward(xfft, index_forward=r)
        else:
            xfft = compress_2D_index_forward_full(xfft, index_forward=r)

        if ctx is not None:
            ctx.xfft = xfft
            if args.is_DC_shift is True:
                ctx.xfft = shift_DC(xfft, onesided=onesided)

        # Correct real coefficients:
        xfft = correct_reals(xfft)

        H_xfft = xfft.shape[-3]
        W_xfft = xfft.shape[-2]
        if onesided:
            W_xfft = (W_xfft - 1) * 2

        out = torch.irfft(input=xfft,
                          signal_ndim=FFTBandFunction2DPool.signal_ndim,
                          signal_sizes=(H_xfft, W_xfft),
                          onesided=onesided)
        return out

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


def correct_real(xfft, y, x):
    """
    Correct a single value to be real.

    :param xfft: input fft map
    :param y: y coordinate
    :param x: x coordinate
    :return: zero out the imaginary part of the (y,x) map
    """
    xfft[..., y, x, 1] = 0

def correct_reals(xfft):
    """
    Correct the 4 values to be real.

    :param xfft:
    :return:
    """
    H_xfft = xfft.shape[-3]
    if H_xfft % 2 == 0:
        even = H_xfft // 2
        xfft = correct_real(xfft, even, 0)
        xfft = correct_real(xfft, even, even)
        xfft = correct_real(xfft, 0, even)
    return xfft


class FFTBand2DPool(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """
    """
    FFT Band layer removes high frequency coefficients.
    """

    def __init__(self, args):
        super(FFTBand2DPool, self).__init__()
        self.args = args

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return FFTBandFunction2DPool.apply(input, self.args)


if __name__ == "__main__":
    print("fft band 2D pool")
