from cnns.nnlib.robustness.channels.channels_definition import compress_svd
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_resize_through_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    to_svd_through_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_through_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    svd_transformation
from cnns.nnlib.utils.general_utils import SVDTransformType


class SVDCompressionTransformation(object):
    """Compress image with SVD.

    Given tensor (C,H,W).

    """

    def __init__(self, args=None, compress_rate=0.0):
        self.args = args
        self.compress_rate = compress_rate
        if args.svd_transform_type == SVDTransformType.STANDARD_TORCH:
            self.compress_f = compress_svd
        elif args.svd_transform_type == SVDTransformType.STANDARD_NUMPY:
            self.compress_f = compress_svd_through_numpy
        elif args.svd_transform_type == SVDTransformType.COMPRESS_RESIZE:
            self.compress_f = compress_svd_resize_through_numpy
        elif args.svd_transform_type == SVDTransformType.TO_SVD_DOMAIN:
            self.args.in_channels = 6
            self.compress_f = to_svd_through_numpy
        elif args.svd_transform_type == SVDTransformType.SYNTHETIC_SVD:
            self.compress_f = svd_transformation
        else:
            raise Exception(
                f'Unknown svd_type: {args.svd_transform_type.name}')

    def __call__(self, data_item):
        """
        Arguments:
            data_item (Tensor): Tensor image of size (C, H, W) to be compressed.

        Returns:
            Tensor: compressed tensor
        """
        return self.compress_f(data_item, compress_rate=self.compress_rate)
