from cnns.nnlib.robustness.channels.channels_definition import fft_channel


class FFTCompressionTransformation(object):
    """Compress image using FFT.

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
        # return compress_svd(data_item, compress_rate=self.compress_rate)
        return fft_channel(data_item.unsqueeze(0),
                           compress_rate=self.compress_rate).squeeze()
