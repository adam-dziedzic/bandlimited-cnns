from cnns.nnlib.robustness.channels.channels_definition import compress_svd
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_resize_through_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    to_svd_through_numpy


class SVDCompressionTransformation(object):
    """Compress image with SVD.

    Given tensor (C,H,W).

    """

    def __init__(self, args=None, compress_rate=0.0):
        self.args = args
        self.compress_rate = compress_rate
        svd_type = 'to_svd_domain'
        if svd_type == 'standard_torch':
            self.compress_f = compress_svd
        elif svd_type == 'compress_resize':
            self.compress_f = compress_svd_resize_through_numpy
        elif svd_type == 'to_svd_domain':
            self.args.in_channels = 6
            self.compress_f = to_svd_through_numpy
        else:
            raise Exception(f'Unknown svd_type: {svd_type}')

    def __call__(self, data_item):
        """
        Arguments:
            data_item (Tensor): Tensor image of size (C, H, W) to be compressed.

        Returns:
            Tensor: compressed tensor
        """
        return self.compress_f(data_item, compress_rate=self.compress_rate)
