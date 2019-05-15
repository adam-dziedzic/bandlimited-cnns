import torch
from torch.nn import Module
from cnns.nnlib.utils.general_utils import next_power2
from torch.nn.functional import pad as torch_pad
import numpy as np


class FFTBandFunction2D(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    signal_ndim = 2

    @staticmethod
    def forward(ctx, input, compress_rate, onesided=True, is_next_power2=False,
                is_test=False):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.

        :param input: the input image
        :param compress_rate: compression rate
        :param onesided: FFT convolution leverages the conjugate symmetry and
        returns only roughly half of the FFT map, otherwise the full map is
        returned
        :param is_next_power2: the FFT is faster if the image size (each side)
        is a power of 2
        :param is_test: test if the number of zero-ed out coefficients is
        correct
        """
        # ctx.save_for_backward(input)
        # print("round forward")
        FFTBandFunction2D.mark_dirty(input)

        N, C, H, W = input.size()

        if H != W:
            raise Exception(f"We support only squared input but the width: {W}"
                            f" is differnt from height: {H}")

        if is_next_power2:
            H_fft = next_power2(H)
            W_fft = next_power2(W)
            pad_H = H_fft - H
            pad_W = W_fft - W
            input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)
        else:
            H_fft = H
            W_fft = W

        xfft = torch.rfft(input, signal_ndim=FFTBandFunction2D.signal_ndim,
                          onesided=onesided)

        del input
        _, _, H_xfft, W_xfft, _ = xfft.size()

        # r - is the side of the retained square in one of the quadrants
        # 4 * r ** 2 / (H * W) = (1 - c)
        # r = np.sqrt((1 - c) * (H * W) / 4)

        compress_rate = compress_rate / 100

        if onesided:
            divisor = 2
        else:
            divisor = 4

        # r - is the length of the side that we retain after compression.
        r = np.sqrt((1 - compress_rate) * H_xfft * W_xfft / divisor)
        # r = np.floor(r)
        r = np.ceil(r)
        r = int(r)

        # zero out high energy coefficients
        if is_test:
            # We divide by 2 to not count zeros complex number twice.
            zero1 = torch.sum(xfft == 0.0).item() / 2
            # print(zero1)

        xfft[..., r:H_fft - r, :, :] = 0.0
        if onesided:
            xfft[..., :, r:, :] = 0.0
        else:
            xfft[..., :, r:W_fft - r, :] = 0.0

        # print(xfft)

        if ctx is not None:
            ctx.xfft = xfft

        if is_test:
            zero2 = torch.sum(xfft == 0.0).item() / 2
            # print(zero2)
            total_size = C * H_xfft * W_xfft
            # print("total size: ", total_size)
            fraction_zeroed = (zero2 - zero1) / total_size
            ctx.fraction_zeroed = fraction_zeroed
            # print("compress rate: ", compress_rate, " fraction of zeroed out: ", fraction_zeroed)
            error = 0.08
            if fraction_zeroed > compress_rate + error or (
                    fraction_zeroed < compress_rate - error):
                raise Exception(f"The compression is wrong, for compression "
                                f"rate {compress_rate}, the number of fraction "
                                f"of zeroed out coefficients "
                                f"is: {fraction_zeroed}")

        out = torch.irfft(input=xfft,
                          signal_ndim=FFTBandFunction2D.signal_ndim,
                          signal_sizes=(H_fft, W_fft),
                          onesided=onesided)
        out = out[..., :H, :W]
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


class FFTBand2D(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """
    """
    FFT Band layer removes high frequency coefficients.
    """

    def __init__(self, args):
        super(FFTBand2D, self).__init__()
        self.compress_rate = args.compress_rate

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return FFTBandFunction2D.apply(input, self.compress_rate)

if __name__ == "__main__":
    compress_rate = 0.6
    H_xfft = 8
    W_xfft = 5
    divisor = 2
    is_test = True

    xfft = np.arange(H_xfft*W_xfft).reshape(H_xfft, W_xfft)
    xfft = torch.tensor(xfft)

    print("initial xfft:\n", xfft)

    r = int(np.sqrt((1 - compress_rate) * H_xfft * W_xfft / divisor))
    print("r: ", r)

    # zero out high energy coefficients
    if is_test:
        # We divide by 2 to not count zeros complex number twice.
        zero1 = torch.sum(xfft == 0.0).item() / 2
        # print(zero1)

    xfft[r:H_xfft - r, :] = 0.0
    xfft[:, r:] = 0.0

    print("compressed xfft:\n", xfft)

    zero2 = torch.sum(xfft == 0.0).item()
    # print(zero2)
    total_size = H_xfft * W_xfft
    # print("total size: ", total_size)
    fraction_zeroed = (zero2 - zero1) / total_size
    print("fraction_zeroed: ", fraction_zeroed)