import torch
from cnns.nnlib.utils.svd2d import compress_svd
from torch.nn import Module


class SVDcompress(Module):
    """
    No PyTorch Autograd used - we compute backward pass on our own.
    """
    """
    SVD compression layer.
    """

    def __init__(self, args):
        super(SVDcompress, self).__init__()
        self.args = args

    def forward(self, input):
        """
        This is the fully manual implementation of the forward and backward
        passes via the torch.autograd.Function.

        :param input: the input map (e.g., an image)
        :return: the result of 1D convolution
        """
        return compress_svd(torch_img=input,
                            compress_rate=self.args.svd_compress)
