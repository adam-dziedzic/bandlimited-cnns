from cnns.nnlib.robustness.channels.channels_definition import compress_svd
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_resize_through_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    to_svd_through_numpy


class SVDCompressionTransformation(object):
    """Compress image with SVD.

    Given tensor (C,H,W).

    """

    def __init__(self, compress_rate=0.0):
        self.compress_rate = compress_rate

    def __call__(self, data_item):
        """
        Arguments:
            data_item (Tensor): Tensor image of size (C, H, W) to be compressed.

        Returns:
            Tensor: compressed tensor
        """
        svd_type = 'to_svd_domain'
        if svd_type == 'standard_torch':
            return compress_svd(data_item, compress_rate=self.compress_rate)
        elif svd_type == 'compress_resize':
            return compress_svd_resize_through_numpy(
                data_item, compress_rate=self.compress_rate)
        elif svd_type == 'to_svd_domain':
            return to_svd_through_numpy(data_item,
                                        compress_rate=self.compress_rate)
