import torch
from torch.nn import Module
from cnns.nnlib.utils.general_utils import next_power2
from torch.nn.functional import pad as torch_pad
from cnns.nnlib.utils.complex_mask import get_disk_mask
from cnns.nnlib.utils.complex_mask import get_hyper_mask


class FFTBandFunctionComplexMask2D(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    signal_ndim = 2

    @staticmethod
    def forward(ctx, input, compress_rate, val=0, interpolate=None,
                get_mask=get_hyper_mask, onesided=True, is_next_power2=False):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.

        :param input: the input image
        :param compress_rate: the compression ratio
        :param val: the value (to change coefficients to) for the mask
        :param interpolate: the interpolation within mask: const, linear, exp,
        log, etc.
        :onesided: should use the onesided FFT thanks to the conjugate symmetry
        or want to preserve all the coefficients
        """
        # ctx.save_for_backward(input)
        # print("round forward")
        FFTBandFunctionComplexMask2D.mark_dirty(input)

        N, C, H, W = input.size()

        if H != W:
            raise Exception("We support only squared input.")

        if is_next_power2:
            H_fft = next_power2(H)
            W_fft = next_power2(W)
            pad_H = H_fft - H
            pad_W = W_fft - W
            input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)
        else:
            H_fft = H
            W_fft = W
        xfft = torch.rfft(input,
                          signal_ndim=FFTBandFunctionComplexMask2D.signal_ndim,
                          onesided=onesided)
        del input
        _, _, H_xfft, W_xfft, _ = xfft.size()
        # assert H_fft == W_xfft, "The input tensor has to be squared."

        mask, _ = get_mask(H=H_xfft, W=W_xfft, compress_rate=compress_rate,
                           val=val, interpolate=interpolate, onesided=onesided)
        mask = mask[:, 0:W_xfft, :]
        # print(mask)
        mask = mask.to(xfft.dtype).to(xfft.device)
        xfft = xfft * mask

        if ctx is not None:
            ctx.xfft = xfft

        out = torch.irfft(input=xfft,
                          signal_ndim=FFTBandFunctionComplexMask2D.signal_ndim,
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
        return grad_output.clone(), None, None, None, None, None, None


class FFTBand2DcomplexMask(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """
    """
    FFT Band layer removes high frequency coefficients.
    """

    def __init__(self, args):
        super(FFTBand2DcomplexMask, self).__init__()
        self.compress_rate = args.compress_rate

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return FFTBandFunctionComplexMask2D.apply(
            input,
            self.compress_rate,
            0, # val
            "exp", # interpolate
            get_hyper_mask, # get_mask
            True, # onesided
            False)  # is nextPower2
