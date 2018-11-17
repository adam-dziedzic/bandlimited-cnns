import torch

class DtypeTransformation(object):
    """Transform a tensor to its dtype representation.

    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image.

        Returns:
            Tensor: with dtype.
        """
        return tensor.to(dtype=self.dtype)