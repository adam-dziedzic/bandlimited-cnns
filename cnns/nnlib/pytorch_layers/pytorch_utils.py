"""
Pytorch utils.

The latest version of Pytorch (in the main branch 2018.06.30) supports tensor flipping.
"""
import torch


def flip(x, dim):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
