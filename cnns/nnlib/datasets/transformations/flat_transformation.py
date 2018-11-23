class FlatTransformation(object):
    """Transform a tensor to its flat representation

    Given tensor (C,H,W), will flatten it to (C, W) - a single data dimension.

    """

    def __call__(self, tensor):
        """
        Arguments:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Flattened tensor (C, W)
        """
        C, H, W = tensor.size()
        tensor = tensor.view(C, H * W)
        return tensor
