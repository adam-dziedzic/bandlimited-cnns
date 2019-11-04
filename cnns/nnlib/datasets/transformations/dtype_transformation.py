import torch


class DtypeTransformation(object):
    """Transform a tensor to its dtype representation.

    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, tensor):
        """
        Arguments:
            tensor (Tensor): Tensor image.

        Returns:
            Tensor: with dtype.
        """
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                tensor[k] = v.to(dtype=self.dtype)
        else:
            tensor = tensor.to(dtype=self.dtype)
        return tensor
