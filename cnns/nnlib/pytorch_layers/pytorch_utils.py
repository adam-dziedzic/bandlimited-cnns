"""
Pytorch track_utils.

The latest version of Pytorch (in the main branch 2018.06.30)
supports tensor flipping.
"""
import collections
import numpy as np
import re
import torch
import torch.nn.functional as F
from torch import tensor
from heapq import heappush, heappop
from cnns.nnlib.utils.general_utils import mem_log_file, next_power2
import gc
from cnns.nnlib.utils.log_utils import get_logger
import logging
import math
import torch_dct

if torch.cuda.is_available():
    # from complex_mul_cpp import complex_mul as complex_mul_cpp
    # from complex_mul_cuda import complex_mul as complex_mul_cuda
    # from complex_mul_cuda import complex_mul_stride as complex_mul_stride_cuda
    from complex_mul_cuda import \
        complex_mul_stride_no_permute as complex_mul_stride_no_permute_cuda
    from complex_mul_cuda import \
        complex_mul_shared_log as complex_mul_shared_log_cuda

logger = get_logger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.info("Set up test")


# Interface to complex multiplication in CUDA.
def complex_mul_shared_log(x, y, out):
    return complex_mul_shared_log_cuda(x, y, out)


def complex_mul_stride_no_permute(x, y, out, num_threads=1024):
    return complex_mul_stride_no_permute_cuda(x, y, out, num_threads)


def get_numpy(x):
    """
    :param x: a tensor from Torch
    :return: a number representation on CPU
    """
    return x.cpu().detach().numpy()


class MockContext(object):
    """
    Mock context class for 1D and 2D convolution. We use it to pass intermediate
    results from the forward to backward pass.
    """

    def __init__(self):
        """
        Set everything to None at the very beginning.
        """
        super(MockContext, self).__init__()
        self.args = None
        self.needs_input_grad = None

    def save_for_backward(self, *args):
        """
        Save intermediate results in the forward pass for the backward pass.
        :param args: the intermediate results to be saved.
        """
        self.__args__ = args

    @property
    def saved_tensors(self):
        """
        Retrieve the saved tensors in the forward pass for the backward pass.
        :return: the saved tensors
        """
        return self.__args__

    def set_needs_input_grad(self, number_needed):
        """
        Set the need for gradients (for the backward pass).

        :param number_needed: how many gradients do we need: for example for the
        input map and the filters.
        """
        self.needs_input_grad = [True for _ in range(number_needed)]


def get_fft_sizes(input_size, filter_size, output_size, padding_count):
    """
    We have to pad input with (filter_size - 1) to execute fft correctly
    (no overlapping signals) and optimize it by extending the signal to the next
    power of 2.
    We want to reuse the fft-ed input x, so we use the larger size chosen from:
    the filter size or output size. Larger padding does not hurt correctness of
    fft but make it slightly slower, in terms of the computation time.

    >>> fft_size, half_fft_size = get_fft_sizes(10, 3, None, 1)
    >>> # print("fft_size: ", fft_size)
    >>> assert fft_size == 32
    >>> assert half_fft_size == 17

    >>> fft_size, half_fft_size = get_fft_sizes(10, 3, 6, 0)
    >>> # print("fft_size: ", fft_size)
    >>> assert fft_size == 16
    >>> assert half_fft_size == 9

    >>> fft_size, half_fft_size = get_fft_sizes(10, 3, 7, 3)
    >>> # print("fft_size: ", fft_size)
    >>> assert fft_size == 32
    >>> assert half_fft_size == 17

    :param input_size: the size of the input for one of the dimensions
    :param filter_size: the size of the filter for one of the dimensions
    :param output_size: the size of the output for one of the dimensions
    :param padding_count: the padding applied to the input for the chosen
    dimension
    :return: the size of the ffted signal (and its onesided size) for the chosen
    dimension
    """
    if output_size is None:
        output_size = input_size - filter_size + 1 + 2 * padding_count
    size = max(filter_size, output_size)
    init_fft_size = next_power2(input_size + size - 1 + 2 * padding_count)
    init_half_fft_size = init_fft_size // 2 + 1
    return init_fft_size, init_half_fft_size


def get_pair(value=None, val_1_default=None, val2_default=None, name="value"):
    """
    >>> v1, v2 = get_pair(9)
    >>> assert v1 == 9
    >>> assert v2 == 9

    >>> value = (8, 1)
    >>> v1, v2 = get_pair(value=value)
    >>> assert v1 == 8
    >>> assert v2 == 1

    >>> v1, v2 = get_pair(val_1_default=3, val2_default=4)
    >>> assert v1 == 3
    >>> assert v2 == 4

    >>> v1, v2 = get_pair((1, 2, 3))
    Traceback (most recent call last):
    ...
    ValueError: value requires a tuple of length 2

    Extend a single value to a 2-element tuple with the same values or just
    return the tuple: value.

    :param value: a number or a tuple
    :param val_1_default: default fist value
    :param val2_default: default second value
    :param name: the name of the value
    :return: the 2-element tuple
    """
    if value is None:
        return val_1_default, val2_default
    if isinstance(value, type(0.0)) or isinstance(value, type(0)):
        return value, value
    elif isinstance(value, type(())) or isinstance(value, type([])):
        if len(value) == 2:
            return value
        else:
            raise ValueError(name + " requires a tuple of length 2")


def to_tensor_item(value):
    """
    Transform from None to -1 or retain the initial value.

    :param value: a value to be changed to an element/item in a tensor
    :return: a number representing the value, tensor with value -1 represents
    the None input
    """
    if value is None:
        value = -1.0
    return value


def to_tensor(value):
    """
    `to_tensor` and `from_tensor` methods are used to transfer intermediate
    results between forward and backward pass of the convolution.

    Transform from None to -1 or retain the initial value
    for transition from a value or None to a tensor.

    :param value: a value to be changed to a tensor
    :return: a tensor representing the value, tensor with value -1 represents
    the None input
    """
    if value is None:
        return tensor([-1])
    return tensor([value])


def from_tensor(tensor_item):
    """
    `to_tensor` and `from_tensor` methods are used to transfer intermediate
    results between forward and backward pass of the convolution.

    Transform from tensor to a single numerical value or None (a tensor with
    value -1 is transformed to None).

    :param tensor_item: tensor with a single value
    :return: a single numerical value extracted from the tensor or None if the
    value is -1
    """
    value = tensor_item.item()
    if value == -1:
        return None
    return value


def flip(x, dim):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


# from memory_profiler import profile
# @profile
def complex_mul(x, y):
    """
    Multiply arrays of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> h
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    """
    # mul = torch.mul
    # add = torch.add
    cat = torch.cat

    # ua = x.narrow(dim=-1, start=0, length=1)
    ua = x[..., :1]
    # ud = x.narrow(-1, 1, 1)
    ud = x[..., 1:]
    # va = y.narrow(-1, 0, 1)
    va = y[..., :1]
    # vb = y.narrow(-1, 1, 1)
    vb = y[..., 1:]
    ub = ua + ud
    uc = ud - ua
    vc = va + vb
    uavc = ua * vc
    # real part of the complex number
    # result_rel = add(uavc, mul(mul(ub, vb), -1))
    result_rel = uavc - ub * vb
    # imaginary part of the complex number
    result_im = uc * va + uavc
    # use the last dimension: dim=-1
    result = cat((result_rel, result_im), dim=-1)
    return result


def complex_mul2(x, y):
    """
    Simply multiplication of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul2(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul2(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul2(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    >>> del x
    >>> del y
    """
    # mul = torch.mul
    # add = torch.add
    cat = torch.cat

    # x = a + bi
    # y = c + di
    # x * y = (ac - bd) + i(ad + bc)
    a = x[..., :1]
    b = x[..., 1:]
    c = y[..., :1]
    d = y[..., 1:]

    # relational part of the complex number
    # result_rel = add(uavc, mul(mul(ub, vb), -1))
    result_rel = a * c - b * d
    # imaginary part of the complex number
    result_im = a * d + b * c
    # use the last dimension: dim=-1
    result = cat(tensors=(result_rel, result_im), dim=-1)
    return result


def complex_mul3(x, y):
    """
    Simply multiplication of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul3(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul3(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul3(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    >>> del x
    >>> del y
    """

    # x = a + bi
    # y = c + di
    # x * y = (ac - bd) + i(ad + bc)
    # a = x[..., :1]
    # b = x[..., 1:]
    # c = y[..., :1]
    # d = y[..., 1:]

    # relational part of the complex number
    # result_rel = add(uavc, mul(mul(ub, vb), -1))
    # result_rel = a * c - b * d
    # imaginary part of the complex number
    # result_im = a * d + b * c
    # use the last dimension: dim=-1
    return torch.cat(tensors=(x[..., :1] * y[..., :1] - x[..., 1:] * y[..., 1:],
                              x[..., 1:] * y[..., :1] + x[..., :1] * y[...,
                                                                     1:]),
                     dim=-1)


def complex_mul4(x, y):
    """
    Simply multiplication of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul4(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul4(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul4(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    >>> del x
    >>> del y
    """
    # mul = torch.mul
    # add = torch.add
    # torch.cat(tensors=(x[..., :1]*y[..., :1]-x[..., 1:]*y[..., 1:],
    #                    x[..., 1:]*y[..., :1]+x[..., :1]*y[..., 1:]))
    out = x * y[..., :1]
    out[..., :1].add_(-1 * x[..., 1:] * y[..., 1:])
    out[..., 1:].add_(x[..., :1] * y[..., 1:])
    return out


def complex_mul5(x, y, out):
    """
    Multiply arrays of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> expect = tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(out, expect)

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> expect = tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(out, expect)

    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> expect = tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> expect = tensor([[-4., 7.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> expect = tensor([[-15., 5.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> expect = tensor([[-15., 5.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> expect = tensor([[[-4., 7.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> expect = tensor([[[-4., 7.], [1., 7.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> expect = tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul5(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)
    """
    # # ua = x.narrow(dim=-1, start=0, length=1)
    # ua = x[..., :1]
    # # ud = x.narrow(-1, 1, 1)
    # ud = x[..., 1:]
    # # va = y.narrow(-1, 0, 1)
    # va = y[..., :1]
    # # vb = y.narrow(-1, 1, 1)
    # vb = y[..., 1:]
    # ub = ua + ud
    # uc = ud - ua
    # vc = va + vb
    # uavc = ua * vc
    # # result_rel = add(uavc, mul(mul(ub, vb), -1))
    # result_rel = uavc - ub * vb
    # # imaginary part of the complex number
    # result_im = uc * va + uavc

    uavc = x[..., 0] * (y[..., 0] + y[..., 1])
    # real part of the complex number
    # result_rel = add(uavc, mul(mul(ub, vb), -1))
    out[..., 0] = uavc - (x[..., 0] + x[..., 1]) * y[..., 1]
    # imaginary part of the complex number
    out[..., 1] = (x[..., 1] - x[..., 0]) * y[..., 0] + uavc


def complex_mul6_cpp(x, y):
    """
    Multiply arrays of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul6_cpp(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul6_cpp(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))
    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = complex_mul6_cpp(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    """
    pass
    # return complex_mul_cpp(x, y)


def complex_mul7_cuda(x, y, out):
    """
    Multiply arrays of complex numbers (it also handles
    multidimensional arrays of complex numbers). Each complex
    number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural
    Networks (https://arxiv.org/pdf/1509.09308.pdf)

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> expect = tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(out, expect)

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> expect = tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(out, expect)

    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> expect = tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> expect = tensor([[-4., 7.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> expect = tensor([[-15., 5.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> expect = tensor([[-15., 5.]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> expect = tensor([[[-4., 7.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> expect = tensor([[[-4., 7.], [1., 7.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)

    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> expect = tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]])
    >>> out = torch.empty_like(expect)
    >>> complex_mul7_cuda(x, y, out)
    >>> np.testing.assert_array_equal(expect, out)
    """
    return complex_mul_cuda(x, y, out)


def complex_mul8_stride_cuda(x, y, out):
    return complex_mul_strid_cuda(x, y, out)


def pytorch_conjugate(x):
    """
    Conjugate all the complex numbers in tensor x (not in place, clone x).

    :param x: PyTorch tensor with complex numbers
    :return: conjugated numbers in x

    >>> x = tensor([[1, 2]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2]]))
    >>> x = tensor([[1, 2], [3, 4]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2], [3, -4]]))
    >>> x = tensor([[[1, 2], [3, 4]], [[0.0, 0.0], [0., 1.]]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x,
    ... tensor([[[1, -2], [3, -4]], [[0., 0.], [0., -1]]]))

    >>> # conjugate 2D
    >>> x = tensor([[1, 2], [3, 4]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2], [3, -4]]))

    >>> # conjugate 2D - these are complex numbers
    >>> x = tensor([[[[1, 2], [3, 4], [5, 6]], [[3,2], [1, 0], [0, 3]],
    ... [[1, 2], [8, 9], [10, 121]]]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[[[1, -2], [3, -4], [5, -6]],
    ... [[3, -2], [1, 0], [0, -3]], [[1, -2], [8, -9], [10, -121]]]]))
    >>> del x
    """
    con_x = x.clone()
    con_x.narrow(dim=-1, start=1, length=1).mul_(-1)
    return con_x


def get_full_energy(x):
    """
    Return the full energy of the signal. The energy E(xfft) of a
    sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/
    sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x and its absolute values (squared)

    >>> x = torch.tensor([1.2, 1.0])
    >>> full_energy, squared = get_full_energy(x)
    >>> np.testing.assert_almost_equal(full_energy, 2.4400, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, torch.tensor([2.4400]))

    >>> x_torch = torch.tensor([[1.2, 1.0], [0.5, 1.4]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    >>> x_torch = torch.tensor([[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)
    """
    # the signal in frequency domain is symmetric and pytorch already
    # discards second half of the signal
    squared = torch.add(torch.pow(x.narrow(-1, 0, 1), 2),
                        torch.pow(x.narrow(-1, 1, 1), 2)).squeeze()
    # sum of squared values of the signal
    full_energy = torch.sum(squared).item()
    return full_energy, squared


def get_phase(x):
    """
    :param x: input complex tensor
    :return: arctan(b/a) for a complex number: z = a + bi

    Get the phase of the complex numbers in the x tensor.

    >>> a = 1.2
    >>> b = -2.3
    >>> x = torch.tensor([a, b])
    >>> phase = get_phase(x)
    >>> np.testing.assert_almost_equal(phase, np.arctan2(b, a), decimal=4)

    """
    return torch.atan2(x.narrow(-1, 1, 1), x.narrow(-1, 0, 1)).squeeze(-1)


def get_full_energy_only(x):
    """
    Return the full energy of the signal. The energy E(xfft) of a
    sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/
    sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x

    >>> x = torch.tensor([1.2, 1.0])
    >>> full_energy = get_full_energy_only(x)
    >>> np.testing.assert_almost_equal(full_energy, 2.4400, decimal=4)

    >>> x_torch = torch.tensor([[1.2, 1.0], [0.5, 1.4]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> # print("x_numpy: ", x_numpy)
    >>> full_energy = get_full_energy_only(x_torch)
    >>> expected_full_energy = np.sum(np.power(np.absolute(x_numpy), 2))
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)

    >>> x_torch = torch.tensor([[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy = get_full_energy_only(x_torch)
    >>> expected_full_energy = np.sum(np.power(np.absolute(x_numpy), 2))
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    """
    return torch.sum(torch.add(torch.pow(x.narrow(-1, 0, 1), 2),
                               torch.pow(x.narrow(-1, 1, 1), 2))).item()


def get_spectrum(xfft):
    """

    :param xfft: input signal in the frequency domain with complex numbers
    represented as an additional dimension with 2 values (real and complex
    parts)
    :return: the squeezed spectrum (sqrt(pow(real_part, 2) +
    pow(complex_part, 2)

    >>> xfft = tensor([[1.0, 3.0], [4.0, 0.0], [-1.0, 2.0]])
    >>> spectrum = get_spectrum(xfft)
    >>> expected = tensor([math.sqrt(10.0), 4.0, math.sqrt(5.0)])
    >>> np.testing.assert_array_almost_equal(x=spectrum, y=expected, err_msg=f"The obtained value {spectrum} is different than the expected value {expected}.")
    """
    squared = torch.add(torch.pow(xfft.narrow(-1, 0, 1), 2),
                        torch.pow(xfft.narrow(-1, 1, 1), 2))
    spectrum = torch.sqrt(squared).squeeze(dim=-1)
    return spectrum


def get_full_energy_simple(x):
    """
    Return the full energy of the signal. The energy E(xfft) of a
    sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/
    sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x

    >>> # Add channels.
    >>> x_torch = torch.tensor([[[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]],
    ... [[-1.0, 2.5], [3.5, -0.1], [1.2, -9.5]]])
    >>> # change the x_torch to a typical numpy array with complex numbers;
    >>> # compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy_simple(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy,
    ... decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared,
    ... decimal=4)

    >>> x = torch.tensor([1.2, 1.0])
    >>> full_energy, squared = get_full_energy_simple(x)
    >>> np.testing.assert_almost_equal(full_energy, 2.4400, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, torch.tensor([2.4400]))

    >>> x_torch = torch.tensor([[1.2, 1.0], [0.5, 1.4]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy_simple(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    >>> x_torch = torch.tensor([[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]])
    >>> # change the x_torch to a typical numpy array with complex numbers;
    >>> # compare the results from numpy and pytorch
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> full_energy, squared = get_full_energy_simple(x_torch)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    """
    # print(x[..., 0])
    # print(x[..., 1])
    # The signal in frequency domain is symmetric and pytorch already
    # discards second half of the signal.
    squared = torch.add(torch.pow(x[..., 0], 2),
                        torch.pow(x[..., 1], 2)).squeeze()
    # sum of squared values of the signal
    full_energy = torch.sum(squared).item()
    return full_energy, squared


def get_full_energy_bulk(x):
    """
    Return the full energy of the signals. The energy E(xfft) of a
    sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/
    sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x

    >>> # Add channels.
    >>> x_torch = torch.tensor([[[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]],
    ... [[-1.0, 2.5], [3.5, -0.1], [1.2, -9.5]]])
    >>> # change the x_torch to a typical numpy array with complex numbers;
    >>> # compare the results from numpy and pytorch
    >>> full_energy, squared = get_full_energy_bulk(x_torch)
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> # print("x_numpy: ", x_numpy)
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared, axis=-1)
    >>> # print("expected_full_energy: ", expected_full_energy)
    >>> expected_full_energy = np.expand_dims(expected_full_energy, axis=1)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy,
    ... decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared,
    ... decimal=4)

    >>> x = torch.tensor([1.2, 1.0])
    >>> full_energy, squared = get_full_energy_bulk(x)
    >>> np.testing.assert_almost_equal(full_energy, 2.4400, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, torch.tensor([2.4400]))

    >>> x_torch = torch.tensor([[1.2, 1.0], [0.5, 1.4]])
    >>> # change the x_torch to a typical numpy array with complex numbers; compare the results from numpy and pytorch
    >>> full_energy, squared = get_full_energy_bulk(x_torch)
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    >>> x_torch = torch.tensor([[-10.0, 1.5], [2.5, 1.8], [1.0, -9.0]])
    >>> # change the x_torch to a typical numpy array with complex numbers;
    >>> # compare the results from numpy and pytorch
    >>> full_energy, squared = get_full_energy_bulk(x_torch)
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> expected_squared = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> expected_full_energy = np.sum(expected_squared)
    >>> np.testing.assert_almost_equal(full_energy, expected_full_energy, decimal=4)
    >>> np.testing.assert_array_almost_equal(squared, expected_squared, decimal=4)

    """
    # print(x[..., 0])
    # print(x[..., 1])
    # The signal in frequency domain is symmetric and pytorch already
    # discards second half of the signal.
    real_squared = torch.pow(x[..., 0], 2)
    img_squared = torch.pow(x[..., 1], 2)
    squared = torch.add(real_squared, img_squared)
    # sum of squared values of the signal
    full_energy = torch.sum(squared, dim=-1, keepdim=True)
    return full_energy, squared


def preserve_energy_index(xfft, energy_rate=None, index_back=None):
    """
    To which index should we preserve the xfft signal (and discard
    the remaining coefficients). This is based on the provided
    energy_rate, or if the energy_rate is not provided, then
    compress_rate is applied.

    :param xfft: the input signal to be truncated
    :param energy_rate: how much energy should we preserve in the xfft signal
    :param index_back: how many coefficients of xfft should we discard counting from the back of the xfft
    :return: calculated index, no truncation applied to xfft itself, returned at least the index of first coefficient
    >>> xfft = torch.tensor([])
    >>> result = preserve_energy_index(xfft, energy_rate=1.0)
    >>> np.testing.assert_equal(result, None)

    >>> xfft = torch.tensor([[1., 2.], [3., 4.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=1.0)
    >>> np.testing.assert_equal(result, 3)

    >>> xfft = [[1., 2.], [3., 4.]]
    >>> result = preserve_energy_index(xfft)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = [[1., 2.], [3., 4.], [5., 6.]]
    >>> result = preserve_energy_index(xfft, compress_rate=1)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = torch.tensor([[1., 2.], [3., 4.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.9)
    >>> np.testing.assert_equal(result, 2)

    >>> xfft = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.5)
    >>> np.testing.assert_equal(result, 1)

    >>> xfft = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> result = preserve_energy_index(xfft, energy_rate=0.0)
    >>> np.testing.assert_equal(result, 1)

    >>> x_torch = torch.tensor([[3., 10.], [1., 1.], [0.1, 0.1]])
    >>> x_numpy = x_torch[...,0].numpy() + 1.0j * x_torch[...,1].numpy()
    >>> squared_numpy = np.power(np.absolute(np.array(x_numpy)), 2)
    >>> full_energy_numpy = np.sum(squared_numpy)
    >>> # set energy rate to preserve only the first and second
    >>> # coefficients
    >>> energy_rate = squared_numpy[0]/full_energy_numpy + 0.0001
    >>> result = preserve_energy_index(x_torch,
    ... energy_rate=energy_rate)
    >>> np.testing.assert_equal(result, 2)
    >>> # set energy rate to preserved only the first coefficient
    >>> energy_rate = squared_numpy[0]/full_energy_numpy - 0.0001
    >>> result = preserve_energy_index(x_torch,
    ... energy_rate=energy_rate)
    >>> np.testing.assert_equal(result, 1)
    """
    if xfft is None or len(xfft) == 0:
        return None
    if energy_rate is not None:
        full_energy, squared = get_full_energy(xfft)
        current_energy = 0.0
        preserved_energy = full_energy * energy_rate
        index = 0
        while current_energy < preserved_energy and index < len(squared):
            current_energy += squared[index]
            index += 1
        return max(index, 1)
    elif index_back is not None:
        return len(xfft) - index_back
    return len(xfft)


def preserve_energy_index_back(xfft, preserve_energy_rate=None,
                               is_reversed=False):
    """
    Give compress_rate for the given energy rate.

    :param xfft: the input fft-ed signal
    :param energy_rate: how much energy of xfft should be preserved?
    :return: the index back (how many coefficient from the end of the signal
    should be discarded?

    >>> xfft = torch.tensor([[
    ... [[5, 6], [3, 4], [1, 2]], [[0, 1], [1, 0], [2, 2]]],
    ... [[[-1, 3], [1, 0], [0, 2]], [[1, 1], [1, -2], [3, 2]]]])
    >>> compress_rate = preserve_energy_index_back(xfft, 50)
    >>> np.testing.assert_equal(compress_rate, 2)

    >>> xfft = torch.tensor([
    ... [ # first signal
    ... [[5, 6], [3, 4], [1, 2]], # first channel
    ... [[0, 1], [1, 0], [2, 2]] # second channel
    ... ],
    ... [ # second signal
    ... [[-1, 3], [1, 0], [0, 2]], # fist channel
    ... [[1, 1], [1, -2], [3, 2]]  # second channel
    ... ]
    ... ])
    >>> compress_rate = preserve_energy_index_back(xfft, 50)
    >>> np.testing.assert_equal(compress_rate, 2)
    """
    # The second dimension from the end is the length because this is a complex
    # signal.
    input_length = xfft.shape[-2]
    if xfft is None or len(xfft) == 0:
        return 0
    squared = torch.add(torch.pow(xfft[..., 0], 2),
                        torch.pow(xfft[..., 1], 2))
    # Sum the batch and channel dimensions (we first reduce to many channels -
    # first 0, and then to only a single channel - next 0 (the dimensions
    # collapse one by one).
    squared = squared.sum(dim=0).sum(dim=0)
    assert len(squared) == input_length
    # Sum of squared values of the signal of length input_length.
    full_energy = torch.sum(squared).item()
    current_energy = 0.0
    preserved_energy = full_energy * preserve_energy_rate / 100.0
    index = 0
    increment = 1
    if is_reversed:
        index = input_length - 1
        increment = -1
    # Accumulate the energy (and increment the index) until the required
    # preserved energy is reached.
    while current_energy < preserved_energy and (
            index < input_length and index >= 0):
        current_energy += squared[index]
        index += increment
    if current_energy < preserved_energy:
        raise AssertionError("We have to accumulate at least preserve energy! "
                             "The index is too low.")
    return input_length - index


def preserve_energy2D_index_back(xfft, preserve_energy_rate=None):
    """
    Give index_back_H and index_back_W for the given energy rate.

    :param xfft: the input fft-ed signal
    :param energy_rate: how much energy of xfft should be preserved?
    :return: the index back for H and W (how many coefficients from both ends of
    the signal should be discarded)?

    Example of an image:
    N=1, C=1, H=3, W=3
    [[
    [[1,2,3],
    [4,5,6],
    [7,8,9]]
    ]]

    The input here is in frequency domain with complex numbers.
    [[
    [[1, 0], [2, 0], [3, 0]],
    [[4, 0], [5,0], [6,0]],
    [[7,0],[8,0],[9,0]]
    ]]

    >>> xfft = tensor([[[[[5, 6], [3, 4], [1, 2]], [[5, 6], [3, 4], [1, 2]]], [[[0, 1], [1, 0], [2, 2]], [[0, 1], [1, 0], [2, 2]]]]])
    >>> np.testing.assert_equal(xfft.size(), [1, 2, 2, 3, 2])
    >>> index_back_H, index_back_W = preserve_energy2D_index_back(xfft, 100)
    >>> np.testing.assert_equal(index_back_H, 0)
    >>> np.testing.assert_equal(index_back_W, 0)

    Test index back for width.
    >>> xfft = torch.tensor([
    ... [ # first image
    ... [[[5, 6], [3, 4], [1, 2]]], # first channel
    ... [[[0, 1], [1, 0], [2, 2]]] # second channel
    ... ],
    ... [ # second image
    ... [[[-1, 3], [1, 0], [0, 2]]], # fist channel
    ... [[[1, 1], [1, -2], [3, 2]]]  # second channel
    ... ]
    ... ])
    >>> index_back_H, index_back_W = preserve_energy2D_index_back(xfft, 50)
    >>> np.testing.assert_equal(index_back_W, 2)
    >>> np.testing.assert_equal(index_back_H, 0)

    Test index back for height.
    >>> xfft = torch.tensor([
    ... [ # first image
    ... [[[5, 6], [3, 4], [1, 2]]], # first channel
    ... [[[0, 1], [1, 0], [2, 2]]] # second channel
    ... ],
    ... [ # second image
    ... [[[-1, 3], [1, 0], [0, 2]]], # fist channel
    ... [[[1, 1], [1, -2], [3, 2]]]  # second channel
    ... ]
    ... ])
    >>> xfft = torch.transpose(xfft, 2, 3)
    >>> index_back_H, index_back_W = preserve_energy2D_index_back(xfft, 50)
    >>> np.testing.assert_equal(index_back_W, 0)
    >>> np.testing.assert_equal(index_back_H, 2)


    >>> xfft = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [2, 5], [3, 0]], # 1st row
    ... [[4, 0], [5, 0], [2, 0]],
    ... [[-1, 1], [1, 0], [1, 0]]]
    ... ]
    ... ])
    >>> index_back_H, index_back_W = preserve_energy2D_index_back(xfft, 50)
    >>> np.testing.assert_equal(index_back_H, 1)
    >>> np.testing.assert_equal(index_back_W, 1)

    >>> xfft = tensor([[[[[5, 6], [3, 4], [1, 2]], [[5, 6], [3, 4], [1, 2]]], [[[0, 1], [1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1]]]]])
    >>> np.testing.assert_equal(xfft.size(), [1, 2, 2, 3, 2])
    >>> index_back_H, index_back_W = preserve_energy2D_index_back(xfft, 60)
    >>> np.testing.assert_equal(index_back_H, 0)
    >>> np.testing.assert_equal(index_back_W, 1)
    """
    if len(xfft.shape) != 5:
        """The dimensions: N, C, H, W, X (complex number)."""
        raise ValueError(
            "The expected input is fft-ed 2D map in a batch with channels.")
    # The third dimension from the end is the length because this is a complex
    # 2D input, the second dimension from the end is the width.
    input_H = xfft.shape[-3]
    input_W = xfft.shape[-2]
    if xfft is None or len(xfft) == 0:
        return 0
    squared = torch.add(torch.pow(xfft[..., 0], 2),
                        torch.pow(xfft[..., 1], 2))
    # Sum the batch and channel dimensions (we first reduce to many channels -
    # first 0, and then to only a single channel - next 0 (the dimensions
    # collapse one by one).
    squared = squared.sum(dim=0).sum(dim=0)
    assert squared.shape[0] == input_H
    assert squared.shape[1] == input_W

    # Sum of squared values of the signal of length input_length.
    full_energy = torch.sum(squared).item()
    current_energy = 0.0
    preserved_energy = full_energy * preserve_energy_rate / 100.0
    index_H = 0
    index_W = 0
    # Accumulate the energy (and increment the index) until the required
    # preserved energy is reached.
    while current_energy < preserved_energy and (
            index_W < input_W and index_H < input_H):
        """
        Try to generate a square output first.
        """
        current_energy += get_squared_energy(squared, index_W)
        index_W += 1
        index_H += 1

    # Then proceed along the columns and then rows.
    while current_energy < preserved_energy and index_W < input_W:
        col_energy = torch.sum(squared[:, index_W]).item()
        current_energy += col_energy
        index_W += 1
    while current_energy < preserved_energy and index_H < input_H:
        row_energy = torch.sum(squared[index_H, :]).item()
        current_energy += row_energy
        index_H += 1
    if current_energy < preserved_energy:
        raise AssertionError("We have to accumulate at least preserve energy! "
                             "The index_H and index_W are too low.")
    return input_H - index_H, input_W - index_W


def zero_out_row_span(xfft, row, start_col, end_col=None):
    """
    Zero out values for all data points, channels, in the given row and column
    span.

    :param xfft: input complex tensor
    :param row: the row number to zero out value
    :param start_col: the start col (inclusive, its value is zeroed out)
    :param end_col: the end col which is not zeroed out (exclusive not zeroed
    out), by default, zero out all the values to the end of the row
    :return: the zero out row for the given col range

    >>> xfft = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [2, 5], [3, 0]], # 1st row
    ... [[4, 0], [5, 0], [2, 0]],
    ... [[-1, 1], [1, 0], [1, 0]]]
    ... ]
    ... ])
    >>> result = zero_out_row_span(xfft, 1, 1, 3)
    >>> expect = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [2, 5], [3, 0]], # 1st row
    ... [[4, 0], [0.0, 0.0 ], [0, 0]],
    ... [[-1, 1], [1, 0], [1, 0]]]
    ... ]
    ... ])
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(result, expect)
    >>> result2 = zero_out_row_span(xfft, 2, 0, 1)
    >>> expect2 = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [2, 5], [3, 0]], # 1st row
    ... [[4, 0], [0.0, 0.0], [0, 0]],
    ... [[0, 0], [1, 0], [1, 0]]]
    ... ]
    ... ])
    >>> np.testing.assert_array_almost_equal(result2, expect2)
    """
    if end_col is None:
        # zero out to the end of the row
        end_col = xfft.shape[3]
    if end_col > start_col:
        xfft[:, :, row, start_col:end_col, :] = torch.zeros(xfft.shape[0],
                                                            xfft.shape[1],
                                                            end_col - start_col,
                                                            xfft.shape[4])
    return xfft


def zero_out_col_span(xfft, col, start_row, end_row=None):
    """
    Zero out values for all data points, channels, in the given column and row
    span.

    :param xfft: input complex tensor
    :param col: the col number to zero out value
    :param start_row: the start row (inclusive, it is zeroed out)
    :param end_row: the end row (exclusive, it is not zeroed out) - by default,
    zero out all the values to the end of the column
    :return: the zero out col for the given row range

    >>> xfft = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [2, 5], [3, 0]], # 1st row
    ... [[4, 0], [5, 0], [2, 0]],
    ... [[-1, 1], [1, 0], [1, 0]]]
    ... ]
    ... ])
    >>> result = zero_out_col_span(xfft, 1, 0, 3)
    >>> expect = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [0, 0], [3, 0]], # 1st row
    ... [[4, 0], [0.0, 0.0 ], [2, 0]],
    ... [[-1, 1], [0, 0], [1, 0]]]
    ... ]
    ... ])
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(result, expect)
    >>> result2 = zero_out_col_span(xfft, 2, 0, 1)
    >>> expect2 = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [0, 0], [0, 0]], # 1st row
    ... [[4, 0], [0.0, 0.0], [2, 0]],
    ... [[-1, 1], [0, 0], [1, 0]]]
    ... ]
    ... ])
    >>> np.testing.assert_array_almost_equal(result2, expect2)
    >>> result3 = zero_out_col_span(xfft, 0, 1, 3)
    >>> expect3 = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[8, 0], [0, 0], [0, 0]], # 1st row
    ... [[0, 0], [0.0, 0.0], [2, 0]],
    ... [[0, 0], [0, 0], [1, 0]]]
    ... ]
    ... ])
    >>> np.testing.assert_array_almost_equal(result3, expect3)
    """
    if end_row is None:
        # zero out to the end of the column
        end_row = xfft.shape[2]
    if end_row > start_row:
        xfft[:, :, start_row:end_row, col, :] = torch.zeros(xfft.shape[0],
                                                            xfft.shape[1],
                                                            end_row - start_row,
                                                            xfft.shape[4])
    return xfft


def preserve_energy2D(xfft, yfft, preserve_energy_rate=None):
    """
        If we zero-out the tail elements from the xfft signal, then the
        corresponding elements from yfft can also be removed since they contribute
        nothing for the element-wise multiplications.

        Given data xfft and filter yfft (in the frequency domain) after adding
        elements one by one to xfft until the
        given preserve_energy_rate is achieved. We add the coefficient becuase
        usually only a few of them are retained from the beginning of the signal,
        and most of them from the tail of the signal are discarded. We also zero-out
        corresponing elements from the filter yfft.

        :param xfft: the input fft-ed signal
        :param yfft: the input fft-ed filter
        :param energy_rate: how much energy of xfft should be preserved?
        :return: xfft and yfft after retaining only the lead coefficients in xfft
        that preserve the given energy rate.

        Example of an image:
        N=1, C=1, H=3, W=3
        [[
        [[1,2,3],
        [4,5,6],
        [7,8,9]]
        ]]

        The input here is in frequency domain with complex numbers.
        [[
        [[1, 0], [2, 0], [3, 0]],
        [[4, 0], [5,0], [6,0]],
        [[7,0],[8,0],[9,0]]
        ]]

        >>> xfft = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[2, 2], [2, 2], [2, 2]], # 1st row
        ... [[2, 2], [2, 2], [-2, -2]],  # 2nd row
        ... [[-2, 2], [-2, 2], [2, 2]]]  # 3rd row
        ... ]
        ... ])
        >>> yfft = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[1, 1], [1, 1], [1, 1]], # 1st row
        ... [[1, 1], [1, 1], [-1, -1]],  # 2nd row
        ... [[-1, 1], [-1, 1], [1, 1]]]  # 3rd row
        ... ]
        ... ])
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft, yfft, preserve_energy_rate=50)
        >>> expect_xfft2 = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[2, 2], [2, 2], [2, 2]], # 1st row
        ... [[2, 2], [2, 2], [0, 0]],  # 2nd row
        ... [[-2, 2], [-2, 2], [0, 0]]]  # 3rd row
        ... ]
        ... ])
        >>> expect_yfft2 = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[1, 1], [1, 1], [1, 1]], # 1st row
        ... [[1, 1], [1, 1], [0, 0]],  # 2nd row
        ... [[-1, 1], [-1, 1], [0, 0]]]  # 3rd row
        ... ]
        ... ])
        >>> np.testing.assert_equal(expect_xfft2.numpy(), xfft.numpy())
        >>> np.testing.assert_equal(expect_yfft2.numpy(), yfft.numpy())
        >>> np.testing.assert_equal(index_back_H, 1)
        >>> np.testing.assert_equal(index_back_W, 0)

        >>> xfft = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[2, 2], [2, 2], [2, 2], [2, 2]], # 1st row
        ... [[2, 2], [2, 2], [-2, -2], [-2, -2]],  # 2nd row
        ... [[-2, 2], [-2, 2], [2, 2], [-2, -2]],  # 3rd row
        ... [[2, 2], [2, 2], [-2, -2], [-2, -2]]]  # 4th row
        ... ]
        ... ])
        >>> yfft = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[1, 1], [1, 1], [1, 1], [1, 1]], # 1st row
        ... [[1, 1], [1, 1], [-1, -1], [1, 1]],  # 2nd row
        ... [[-1, 1], [-1, 1], [1, 1], [1, 1]],  # 3rd row
        ... [[-1, 1], [-1, 1], [1, 1], [1, 1]]]  # 4th row
        ... ]
        ... ])
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft, yfft, preserve_energy_rate=40)
        >>> # print("result xfft2: ", xfft2)
        >>> expect_xfft2 = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[2, 2], [2, 2], [2, 2], [2, 2]], # 1st row
        ... [[2, 2], [2, 2], [-2, -2], [-2, -2]],  # 2nd row
        ... [[-2, 2], [0, 0], [0, 0], [-2, -2]],  # 3rd row
        ... [[2, 2], [2, 2], [-2, -2], [-2, -2]]]  # 4th row
        ... ]
        ... ])
        >>> expect_yfft2 = torch.tensor(
        ... [ # 1 image
        ... [ # 1 channel
        ... [[[1, 1], [1, 1], [1, 1], [1, 1]], # 1st row
        ... [[1, 1], [1, 1], [-1, -1], [1, 1]],  # 2nd row
        ... [[-1, 1], [0, 0], [0, 0], [1, 1]],  # 3rd row
        ... [[-1, 1], [-1, 1], [1, 1], [1, 1]]]  # 4th row
        ... ]
        ... ])
        >>> np.testing.assert_equal(expect_xfft2.numpy(), xfft.numpy())
        >>> np.testing.assert_equal(expect_yfft2.numpy(), yfft.numpy())
        >>> np.testing.assert_equal(index_back_H, 1)
        >>> np.testing.assert_equal(index_back_W, 1)

        >>> xfft = tensor([[[[[5, 6], [3, 4], [1, 2]], [[5, 6], [3, 4], [1, 2]]], [[[0, 1], [1, 0], [2, 2]], [[0, 1], [1, 0], [2, 2]]]]])
        >>> yfft = tensor([[[[[5, 6], [3, 4], [1, 2]], [[5, 6], [3, 4], [1, 2]]], [[[0, 1], [1, 0], [2, 2]], [[0, 1], [1, 0], [2, 2]]]]])
        >>> np.testing.assert_equal(xfft.size(), [1, 2, 2, 3, 2])
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft.clone(), yfft.clone(), 100)
        >>> np.testing.assert_equal(xfft2.numpy(), xfft.numpy())
        >>> np.testing.assert_equal(yfft2.numpy(), yfft.numpy())
        >>> np.testing.assert_equal(index_back_H, 0)
        >>> np.testing.assert_equal(index_back_W, 0)

        Test index back for width.
        >>> xfft = torch.tensor([
        ... [ # first image
        ... [[[2, 2], [2, 3], [2, 2]]], # first channel
        ... [[[2, 2], [2, 2], [2, 2]]] # second channel
        ... ],
        ... [ # second image
        ... [[[1, 1], [1, 2], [1, 1]]], # fist channel
        ... [[[1, 1], [1, 1], [1, 1]]]  # second channel
        ... ]
        ... ], dtype=torch.float)
        >>> yfft = xfft
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft, yfft, 50)
        >>> expect = torch.tensor([
        ... [ # first image
        ... [[[2, 2], [2, 3], [2, 2]]], # first channel
        ... [[[2, 2], [2, 2], [2, 2]]] # second channel
        ... ],
        ... [ # second image
        ... [[[1, 1], [1, 2], [1, 1]]], # fist channel
        ... [[[1, 1], [1, 1], [1, 1]]]  # second channel
        ... ]
        ... ], dtype=torch.float)
        >>> # print("obtained xfft2: ", xfft2)
        >>> np.testing.assert_equal(xfft2.numpy(), expect.numpy())
        >>> np.testing.assert_equal(yfft2.numpy(), expect.numpy())
        >>> np.testing.assert_equal(index_back_H, 0)
        >>> np.testing.assert_equal(index_back_W, 1)

        Test index back for height.
        >>> xfft = torch.tensor([
        ... [ # first image
        ... [[[2, 2], [2, 3], [2, 2]]], # first channel
        ... [[[2, 2], [2, 2], [2, 2]]] # second channel
        ... ],
        ... [ # second image
        ... [[[1, 1], [1, 2], [1, 1]]], # fist channel
        ... [[[1, 1], [1, 1], [1, 1]]]  # second channel
        ... ]
        ... ])
        >>> xfft = torch.transpose(xfft, 2, 3)
        >>> yfft = xfft
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft.clone(), yfft.clone(), 50)
        >>> np.testing.assert_equal(xfft2.numpy(), xfft.numpy())
        >>> np.testing.assert_equal(yfft2.numpy(), xfft.numpy())
        >>> np.testing.assert_equal(index_back_H, 1)
        >>> np.testing.assert_equal(index_back_W, 0)

        >>> xfft = tensor([[[  # 2 channels, 2 x 3 images
        ... [[5, 6], [3, 4], [1, 2]],
        ... [[5, 6], [3, 4], [1, 2]]],
        ... [[[0, 1], [1, 0], [0, 1]],
        ... [[0, 1], [1, 0], [0, 1]]]]])
        >>> squared = torch.add(torch.pow(xfft[..., 0], 2), torch.pow(xfft[..., 1], 2))
        >>> squared = squared.sum(dim=0).sum(dim=0)
        >>> expect_squared = tensor([  # 2 x 3 map
        ... [62, 26, 6],
        ... [62, 26, 6]])
        >>> np.testing.assert_equal(expect_squared.numpy(), squared.numpy())
        >>> np.testing.assert_equal(xfft.size(), [1, 2, 2, 3, 2])
        >>> yfft = xfft
        >>> xfft2, yfft2, index_back_H, index_back_W = preserve_energy2D(xfft.clone(), yfft.clone(), 60)
        >>> expect_xfft = tensor([[[  # 2 channels, 2 x 3 images
        ... [[5, 6], [3, 4], [1, 2]],
        ... [[5, 6], [0, 0], [1, 2]]],
        ... [[[0, 1], [1, 0], [0, 1]],
        ... [[0, 1], [0, 0], [0, 1]]]]])
        >>> expect_yfft = expect_xfft
        >>> np.testing.assert_equal(xfft2.numpy(), expect_xfft.numpy())
        >>> np.testing.assert_equal(yfft2.numpy(), expect_yfft.numpy())
        >>> np.testing.assert_equal(index_back_H, 0)
        >>> np.testing.assert_equal(index_back_W, 1)
    """
    if len(xfft.shape) != 5:
        """The dimensions: N, C, H, W, X (complex number)."""
        raise ValueError(
            "The expected input is fft-ed 2D map in a batch with channels.")
    # The second dimension from the end is the length because this is a complex
    # 2D input.
    input_H = xfft.shape[2]
    input_W = xfft.shape[3]
    if xfft is None or len(xfft) == 0:
        return xfft, yfft, 0, 0
    squared = torch.add(torch.pow(xfft[..., 0], 2),
                        torch.pow(xfft[..., 1], 2))
    # Sum the batch and channel dimensions (we first reduce to many channels -
    # first 0, and then to only a single channel - next 0 (the dimensions
    # collapse one by one).
    squared = squared.sum(dim=0).sum(dim=0)
    assert squared.shape[0] == input_H
    assert squared.shape[1] == input_W

    # Sum of squared values of the signal of length input_length.
    full_energy = torch.sum(squared).item()
    current_energy = 0.0
    preserved_energy = full_energy * preserve_energy_rate / 100.0
    # iterate vertically
    col = 0  # iteration column
    col_count = input_W  # for code readability
    row_index = 0  # current vertical iteration index
    # first count cell in column then in row - this is needed to prevent
    # counting a given squared (energy) cell twice
    old_col_index = row_index
    # iterate horizontally
    row = 0  # try not to overlap the row and col cells
    row_count = input_H  # for code readability
    col_index = 0  # current horizontal iteration index
    # Accumulate the energy (and increment the indexes) until the required
    # preserved energy is reached.
    # print("col count: ", col_count)
    # print("row_count: ", row_count)
    while current_energy < preserved_energy:
        if (row_index == col) and (col_index == row):
            # Add the corner value (and we need at least one coefficient).
            current_energy += squared[row_index][col_index]
            if col < col_count:
                col += 1
                if col < col_count:
                    row_index = 0
                # otherwise, keep the row_index as a sentinel for the row traversal
            if row < row_count:
                row += 1
                if row < row_count:
                    col_index = 0
                # otherwise: keep the col_index as a sentinel for the col traversal
        # we have more rows than columns in the energy matrix squared
        elif col_index == col and row < row_count and row_count > col_count:
            # move to the next row
            row += 1
            col_index = 0
        # we have more columns than rows in the energy matrix squared
        elif row_index == row and col < col_count and col_count > row_count:
            # move to the next column
            col += 1
            row_index = 0
        # iterate through the current column
        if col < col_count and current_energy < preserved_energy and (
                row_index < row):
            # print("col: ", col)
            # print("row_index: ", row_index)
            current_energy += squared[row_index][col]
            row_index += 1
        if row < row_count and current_energy < preserved_energy and (
                col_index < col):
            # print("row: ", row)
            # print("col_index: ", col_index)
            current_energy += squared[row][col_index]
            col_index += 1

    if current_energy < preserved_energy:
        raise AssertionError(f"We have to accumulate at least preserve energy: "
                             f"{preserved_energy}. Accumulated energy is: "
                             f"{current_energy}.")

    index_back_W = 0
    if col < col_count:
        if row_index == 0:
            # the whole column should be discarded
            index_back_W = col_count - col
        else:
            index_back_W = (col_count - 1) - col
            # zero out part of the column
            end_row = min(row + 1, row_count)
            xfft = zero_out_col_span(xfft, col=col, start_row=row_index,
                                     end_row=end_row)
            yfft = zero_out_col_span(yfft, col=col, start_row=row_index,
                                     end_row=end_row)

    index_back_H = 0
    if row < row_count:
        if col_index == 0:
            # the whole row should be discarded
            index_back_H = row_count - row
        else:
            index_back_H = (row_count - 1) - row
            # zero out part of the row
            end_col = min(col + 1, col_count)
            xfft = zero_out_row_span(xfft, row=row, start_col=col_index,
                                     end_col=end_col)
            yfft = zero_out_row_span(yfft, row=row, start_col=col_index,
                                     end_col=end_col)

    return xfft, yfft, index_back_H, index_back_W


def preserve_energy2D_symmetry(xfft, yfft, preserve_energy_rate=None,
                               is_debug=False):
    """
    Compress xfft and yfft taking into account Hermitian symmetry of the fft-ed
    2D maps.

    :param xfft: input activation map in frequency domain.
    :param yfft: filter in frequency domain.
    :param preserve_energy_rate: how much energy to preserve in the input
    activation map.
    :return: the compressed xfft and yfft.

    >>> # xfft: tensor 6 x 4 (4 = 6 // 2 + 1)
    >>> xfft = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[2, 2], [2, 2], [2, 2], [2, 2]], # 1st row
    ... [[2, 2], [2, 2], [-2, -2], [-2, -2]],  # 2nd row
    ... [[-2, 2], [-2, 2], [2, 2], [-2, -2]],  # 3rd row
    ... [[2, 2], [2, 2], [-2, -2], [-2, -2]],
    ... [[2, 2], [2, 2], [-2, -2], [-2, -2]],
    ... [[2, 2], [2, 2], [-2, -2], [-2, -2]]]  # 6th row
    ... ]
    ... ])
    >>> yfft = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[1, 1], [1, 1], [1, 1], [1, 1]],  # 1st row
    ... [[1, 1], [1, 1], [-1, -1], [1, 1]], # 2nd row
    ... [[-1, 1], [-1, 1], [1, 1], [1, 1]],
    ... [[-1, 1], [-1, 1], [1, 1], [1, 1]],
    ... [[-1, 1], [-1, 1], [1, 1], [1, 1]],
    ... [[-1, 1], [-1, 1], [1, 1], [1, 1]]]
    ... ]
    ... ])
    >>> xfft2, yfft2 = preserve_energy2D_symmetry(xfft, yfft, preserve_energy_rate=40)
    >>> # print("result xfft2: ", xfft2)
    >>> expect_xfft2 = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[2, 2], [2, 2], [2, 2]], # 1st row
    ... [[2, 2], [2, 2], [-2, -2]],  # 2nd row
    ... [[-2, 2], [-2, 2], [2, 2]],  # 3rd row
    ... [[2, 2], [2, 2], [-2, -2]],
    ... [[2, 2], [2, 2], [-2, -2]]]  # 6th row
    ... ]
    ... ])
    >>> expect_yfft2 = torch.tensor(
    ... [ # 1 image
    ... [ # 1 channel
    ... [[[1, 1], [1, 1], [1, 1]],  # 1st row
    ... [[1, 1], [1, 1], [-1, -1]], # 2nd row
    ... [[-1, 1], [-1, 1], [1, 1]],
    ... [[-1, 1], [-1, 1], [1, 1]],
    ... [[-1, 1], [-1, 1], [1, 1]]]
    ... ]
    ... ])
    >>> np.testing.assert_equal(xfft2.numpy(), expect_xfft2.numpy())
    >>> np.testing.assert_equal(yfft2.numpy(), expect_yfft2.numpy())
    """
    if xfft is None or len(xfft) == 0:
        return xfft, yfft
    if preserve_energy_rate == 100.0:
        return xfft, yfft
    if preserve_energy_rate >= 100.0:
        raise Exception("preserve_energy_rate should in % from 0.0 to 100.0.")
    if len(xfft.shape) != 5:
        raise ValueError(
            "The expected input is fft-ed 2D map in a batch with channels. "
            "The expected dimensions: N, C, H, W, X (complex number)")
    input_W = xfft.shape[3]
    preserved_energy = get_full_energy_only(xfft) * preserve_energy_rate / 100.0

    # binary search
    low = 0  # lower bound inclusive
    high = input_W  # upper bound inclusive
    index = low + (high - low) // 2
    while low < high:
        if compress_2D_energy(xfft, index_forward=index) >= preserved_energy:
            high = index
            index = low + (high - low) // 2
        else:
            low = index + 1
            index = low + (high - low) // 2

    cxfft = compress_2D_index_forward(xfft, index)
    cyfft = compress_2D_index_forward(yfft, index)

    # if is_debug:
    #     xfft_numel = xfft.numel()
    #     cxfft_numel = cxfft.numel()
    #     compression_ratio = (xfft_numel - cxfft_numel) / xfft_numel
    #     print(f"total width,{input_W},index forward,{index}, "
    #           f"num elems xfft,{xfft_numel}, "
    #           f"num elems compressed xfft,{cxfft_numel}, "
    #           f"compression ratio,{compression_ratio},stop")
    return cxfft, cyfft


def compress_2D_energy(xfft, index_forward):
    """
    Return energy for compress xfft to index_forward coefficients.

    :param xfft: input xfft-ed image/filter
    :param index_forward: how many indexes are preserved (this serves as size
    of the side of the squares preserved in top left and bottom left part of
    xfft).
    :return: the energy preserved

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    >>> index_forward = 3
    >>> xfft_compressed = compress_2D_index_forward(xfft, index_forward = index_forward)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print("xfft: ", xfft)
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 3
    >>> energy_expected = get_full_energy_only(xfft_compressed)
    >>> energy_computed = compress_2D_energy(xfft, index_forward = index_forward)
    >>> np.testing.assert_approx_equal(actual=energy_computed, desired=energy_expected)

    """
    n = index_forward - 1
    if n < 0:
        return 0.0
    energy_top_left = get_full_energy_only(xfft[:, :, :n + 1, :n + 1, :])
    if n > 0:
        energy_bottom_left = get_full_energy_only(xfft[:, :, -n:, :n + 1, :])
        return energy_top_left + energy_bottom_left
    else:
        return energy_top_left


def compress_2D_index_back(xfft, index_back):
    half_W_fft = xfft.shape[-2]
    index_forward = get_index_forward(half_W_fft, index_back)
    return compress_2D_index_forward(xfft, index_forward)


def compress_2D_index_forward(xfft, index_forward):
    """
    Compress xfft to index_forward coefficients with odd index_forward param.

    :param xfft: input xfft-ed image/filter
    :param index_forward: how many indexes are preserved (this serves as size
    of the side of the squares preserved in top left and bottom left part of
    xfft).
    :return: the compressed signal

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    >>> # print("xfft: ", xfft)
    >>> xfft_compressed = compress_2D_index_forward(xfft, index_forward = 3)
    >>> # print("xfft compressed: ", xfft_compressed)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 3

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    >>> # print("xfft: ", xfft)
    >>> xfft_compressed = compress_2D_index_forward(xfft, index_forward = 4)
    >>> # print("xfft compressed: ", xfft_compressed)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 6
    >>> assert W == 4

    """
    if index_forward == xfft.shape[-2]:
        return xfft
    n = index_forward - 1
    top_left = xfft[..., :n + 1, :n + 1, :]
    if n > 0:
        bottom_left = xfft[..., -n:, :n + 1, :]
        return torch.cat((top_left, bottom_left), dim=-3)
    else:
        return top_left


def compress_2D_index_back_full(xfft, index_back):
    W_fft = xfft.shape[-2]
    index_forward = get_index_forward_full(W_fft, index_back)
    return compress_2D_index_forward_full(xfft, index_forward)


def compress_2D_index_forward_full(xfft, index_forward):
    """
    Compress xfft to index_forward coefficients with odd index_forward param.

    :param xfft: input xfft-ed image/filter
    :param index_forward: how many indexes are preserved (this serves as size
    of the side of the squares preserved in top left and bottom left part of
    xfft).
    :return: the compressed signal

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    >>> xfft_compressed = compress_2D_index_forward_full(xfft, index_forward = 2)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 3
    >>> assert W == 3

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    >>> xfft_compressed = compress_2D_index_forward_full(xfft, index_forward = 3)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 5

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    >>> xfft_compressed = compress_2D_index_forward_full(xfft, index_forward = 4)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 6
    >>> assert W == 6

    """
    if index_forward == xfft.shape[-2] // 2 + 1:
        return xfft

    n = index_forward - 1
    top_left = xfft[:, :, :n + 1, :n + 1, :]
    if n > 0:
        bottom_left = xfft[:, :, -n:, :n + 1, :]
        top_right = xfft[:, :, :n + 1, -n:, :]
        bottom_right = xfft[:, :, -n:, -n:, :]

        # Combine along the H - height (vertical dimension).
        left = torch.cat((top_left, bottom_left), dim=2)
        right = torch.cat((top_right, bottom_right), dim=2)
        # Combine along the W - width (horizontal dimension).
        result = torch.cat((left, right), dim=3)
        return result
    else:
        # Return just a single coefficient.
        return top_left


def get_index_forward(W, index_back):
    return W - index_back


def get_index_back(W, index_forward):
    return W - index_forward


def get_index_forward_full(W, index_back):
    return (W // 2 + 1) - index_back


def get_index_back_full(W, index_forward):
    return (W // 2 + 1) - index_forward


def restore_size_2D_batch(xfft, init_H_fft, init_half_W_fft):
    """
    Fill in with zeros to the size initH x initW.

    :param xfft: compressed signal.
    :param init_H_fft: initial height.
    :param init_half_W_fft: initial width.
    :param index_forward: xfft was compressed to the index_forward.
    :return: xfft filled in with zeros to the size initH x initW.
[
    >>> x = tensor([[[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    >>> N, K, C, init_H_fft, init_half_W_fft, _ = xfft.size()
    >>> index_forward = 3
    >>> xfft_compressed = compress_2D_index_forward(xfft, index_forward)
    >>> _, _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 3
    >>> xfft_restored = restore_size_2D_batch(xfft, init_H_fft = init_H_fft, init_half_W_fft=init_half_W_fft)
    >>> NN, KK, CC, initHH, initWW, _ = xfft_restored.size()
    >>> assert initHH == init_H_fft
    >>> assert initWW == init_half_W_fft
    """
    index_forward = xfft.shape[-2]
    if index_forward == init_half_W_fft:
        return xfft
    n = index_forward - 1
    N, K, C, H, W, _ = xfft.size()
    top_left = xfft[..., :n + 1, :n + 1, :]
    if n > 0:
        bottom_left = xfft[..., -n:, :n + 1, :]
        middle = torch.zeros(N, K, C, init_H_fft - (2 * n + 1), n + 1, 2,
                             dtype=xfft.dtype)
        left = torch.cat((top_left, middle, bottom_left), dim=-3)
        right = torch.zeros(N, K, C, init_H_fft, init_half_W_fft - (n + 1), 2,
                            dtype=xfft.dtype)
        result = torch.cat((left, right), dim=-2)
        return result
    else:
        # Return just a single coefficient.
        row = torch.cat(
            (top_left, torch.zeros(N, K, C, 1, init_half_W_fft - 1, 2)), dim=-2)
        result = torch.cat((row,
                            torch.zeros(N, K, C, init_H_fft - 1,
                                        init_half_W_fft,
                                        2)), dim=-3)
        return result


def restore_size_2D(xfft, init_H_fft, init_half_W_fft):
    """
    Fill in with zeros to the size initH x initW.

    :param xfft: compressed signal.
    :param init_H_fft: initial height.
    :param init_half_W_fft: initial width.
    :param index_forward: xfft was compressed to the index_forward.
    :return: xfft filled in with zeros to the size initH x initW.

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    >>> N, C, init_H_fft, init_half_W_fft, _ = xfft.size()
    >>> index_forward = 3
    >>> xfft_compressed = compress_2D_index_forward(xfft, index_forward)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 3
    >>> xfft_restored = restore_size_2D(xfft, init_H_fft = init_H_fft, init_half_W_fft=init_half_W_fft)
    >>> NN, CC, initHH, initWW, _ = xfft_restored.size()
    >>> assert initHH == init_H_fft
    >>> assert initWW == init_half_W_fft
    """
    index_forward = xfft.shape[-2]
    if index_forward == init_half_W_fft:
        return xfft
    n = index_forward - 1
    N, C, H, W, _ = xfft.size()
    top_left = xfft[..., :n + 1, :n + 1, :]
    if n > 0:
        bottom_left = xfft[..., -n:, :n + 1, :]
        middle = torch.zeros(N, C, init_H_fft - (2 * n + 1), n + 1, 2,
                             dtype=xfft.dtype, device=xfft.device)
        left = torch.cat((top_left, middle, bottom_left), dim=2)
        right = torch.zeros(N, C, init_H_fft, init_half_W_fft - (n + 1), 2,
                            dtype=xfft.dtype, device=xfft.device)
        result = torch.cat((left, right), dim=3)
        return result
    else:
        # Return just a single coefficient.
        row = torch.cat(
            (top_left, torch.zeros(N, C, 1, init_half_W_fft - 1, 2,
                                   device=xfft.device, dtype=xfft.dtype)),
            dim=3)
        result = torch.cat((row,
                            torch.zeros(N, C, init_H_fft - 1, init_half_W_fft,
                                        2, device=xfft.device,
                                        dtype=xfft.dtype)), dim=2)
        return result


def restore_size_2D_full(xfft, init_H_fft, init_W_fft):
    """
    Fill in with zeros to the size initH x initW.

    :param xfft: compressed signal.
    :param init_H_fft: initial height.
    :param init_W_fft: initial width.
    :param index_forward: xfft was compressed to the index_forward.
    :return: xfft filled in with zeros to the size initH x initW.

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    >>> N, C, initH, initW, _ = xfft.size()
    >>> assert initH == initW
    >>> index_forward = 3
    >>> xfft_compressed = compress_2D_index_forward_full(xfft, index_forward)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 5
    >>> assert W == 5
    >>> xfft_restored = restore_size_2D_full(xfft, initH, initW)
    >>> NN, CC, initHH, initWW, _ = xfft_restored.size()
    >>> assert initHH == initH
    >>> assert initWW == initW

    >>> x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0, 5.0],
    ... [3.0, 4.0, 1.0, -1.0, 3.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0],
    ... [5.0, 3.0, 0.0, -1.0, 0.0, 4.0],
    ... [3.0, 0.0, 1.0, -1.0, 0.0  , 5.0],
    ... [1.0, 2.0, 1.0, 1.0, 0.0   , 5.0]]]])
    >>> xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    >>> N, C, initH, initW, _ = xfft.size()
    >>> assert initH == initW
    >>> index_forward = 2
    >>> xfft_compressed = compress_2D_index_forward_full(xfft, index_forward)
    >>> _, _, H, W, _ = xfft_compressed.size()
    >>> # print(xfft_compressed.size())
    >>> assert H == 3
    >>> assert W == 3
    >>> xfft_restored = restore_size_2D_full(xfft, initH, initW)
    >>> NN, CC, initHH, initWW, _ = xfft_restored.size()
    >>> assert initHH == initH
    >>> assert initWW == initW
    """
    if xfft.shape[-2] == init_W_fft:
        return xfft
    index_forward = xfft.shape[-2] // 2 + 1
    n = index_forward - 1
    N, C, H, W, _ = xfft.size()
    top_left = xfft[:, :, :n + 1, :n + 1, :]
    if n > 0:
        bottom_left = xfft[:, :, -n:, :n + 1, :]
        middle_left = torch.zeros(N, C, init_H_fft - (2 * n + 1), n + 1, 2)
        left = torch.cat((top_left, middle_left, bottom_left), dim=2)

        top_right = xfft[:, :, :n + 1, -n:, :]
        bottom_right = xfft[:, :, -n:, -n:, :]
        middle_right = torch.zeros(N, C, init_H_fft - (2 * n + 1), n, 2)
        right = torch.cat((top_right, middle_right, bottom_right), dim=2)

        middle = torch.zeros(N, C, init_H_fft, init_W_fft - (2 * n + 1), 2)

        result = torch.cat((left, middle, right), dim=3)
        return result
    else:
        row = torch.cat((top_left, torch.zeros(N, C, 1, W - 1, 2)), dim=3)
        result = torch.cat(row,
                           torch.zerso(N, C, init_H_fft - 1, init_W_fft, 2),
                           dim=2)
        return result


def compress_2D_half_test(x, index_back=0):
    """
    Return a compressed version of x, after its compression in the frequency
    domain.

    :param x: the input 2D image.
    :param index_back: how many indexes are cut-off from the half-ed signal in
    the frequency domain?
    :return: the compressed x after its compression in the frequency domain.
    """
    N, C, H, W = x.size()
    xfft = torch.rfft(x, signal_ndim=2, onesided=True)
    N, C, init_H_fft, init_half_W_fft, _ = xfft.size()
    cxfft = compress_2D_index_back(xfft, index_back)
    cxfft_zeros = restore_size_2D(cxfft, init_H_fft=init_H_fft,
                                  init_half_W_fft=init_half_W_fft)
    cx = torch.irfft(cxfft_zeros, signal_ndim=2, signal_sizes=(H, W),
                     onesided=True)
    return cx


def show2D_spectra_test(x, index_back=0):
    """
    Return a spectral representation of x.

    :param x: the input 2D image.
    :param index_back: how many indexes are cut-off from the half-ed signal in
    the frequency domain?
    :return: the 2D spectrum of x.
    """
    xfft = torch.rfft(x, signal_ndim=2, onesided=False)
    N, C, init_H_fft, init_W_fft, _ = xfft.size()
    cxfft = compress_2D_in(xfft, index_back)
    cxfft_zeros = restore_size_2D_full(cxfft, init_H_fft=init_H_fft,
                                       init_W_fft=init_W_fft)
    return cxfft_zeros


def get_squared_energy(squared, index):
    """
    Get the energy from the next flipped L slice of the squared matrix.

    :param index: the index into col and row
    :return: the energy from the next slice of the squared matrix.

    >>> squared = torch.tensor(
    ... [[1,2,3,4],
    ... [5,6,7,8],
    ... [9,10,11,12],
    ... [13,14,15,16]])
    >>> index = 2
    >>> row_energy =  9 + 10
    >>> col_energy = 3 + 7
    >>> corner_energy = 11
    >>> total_energy = row_energy + col_energy + corner_energy
    >>> assert get_squared_energy(squared, 2) == total_energy
    >>> assert get_squared_energy(squared, 0) == 1
    >>> index_1_energy = 2 + 5 + 6
    >>> assert get_squared_energy(squared, 1) == index_1_energy
    """
    row_energy = torch.sum(squared[index, 0:index]).item()
    col_energy = torch.sum(squared[0:index, index]).item()
    corner_energy = squared[index, index].item()
    total_energy = row_energy + col_energy + corner_energy
    return total_energy


def correlate_signals(x, y, fft_size, out_size, preserve_energy_rate=None,
                      index_back=None, signal_ndim=1):
    """
    Cross-correlation of the signals: x and y.
    Theory: X(f) = fft(x(t)). The first sample X(0) of the
    transformed series is the DC component, more commonly known
    as the average of the input series. For the normalized fft
    (both sums are multiplied by $\frac{1}{\sqrt{N}}$. The length of the sin(x)
    is || sin(x)||^2 = \integral_{0}^{2\pi} sin^2(x) dx = \pi, so
    ||sin(x)||=\sqrt(\pi)

    $$
    \begin{align}
        X(0) = \frac{1}{\sqrt{N}} \sum_{n=0}^{n=N-1} x(n)
    \end{align}
    $$

    :param x: input signal
    :param y: filter
    :param fft_size: the size of the signal in the frequency domain
    :param out_size: required output len (size)
    :param preserve_energy_rate: compressed to this energy rate
    :param index_back: how many coefficients to remove
    :param signal_ndim: what is the dimension of the input data
    :return: output signal after correlation of signals x and y

    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,0.0,4.0,4.0]]])
    >>> # x.shape is torch.Size([1, 2, 4])
    >>> # two filters
    >>> y = tensor([[[1.0,3.0], [1.0,3.0]], [[2.0,1.0], [0.0,1.0]]])
    >>> # y.shape is torch.Size([2, 2, 2])
    >>> # W - width of input signals (time-series)
    >>> W = x.shape[-1]
    >>> # WW - width of the filter
    >>> WW = y.shape[-1]
    >>> out_size = W - WW + 1
    >>> result = correlate_signals(x=x, y=y, fft_size=W, out_size=out_size)
    >>> # print("result: ", result)
    >>> expected = np.array([[[7.0, 11.0, 15.0], [1.0, 12.0, 16.0]],
    ... [[4.0, 7.0, 10.0], [0.0, 4.0, 4.0]]])
    >>> np.testing.assert_array_almost_equal(result, expected)

    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,2.0,3.0,4.0]]])
    >>> y = tensor([[[1.0,3.0], [1.0,3.0]]])
    >>> result = correlate_signals(x=x, y=y, fft_size=x.shape[-1],
    ... out_size=(x.shape[-1]-y.shape[-1] + 1))
    >>> np.testing.assert_array_almost_equal(result,
    ... np.array([[[7.0, 11.0, 15.0], [7.0, 11.0, 15.0]]]))

    >>> x = tensor([1.0,2.0,3.0,4.0])
    >>> y = tensor([1.0,3.0])
    >>> result = correlate_signals(x=x, y=y, fft_size=len(x), out_size=(len(x)-len(y) + 1))
    >>> expected_result = np.correlate(x, y, mode='valid')
    >>> np.testing.assert_array_almost_equal(result, expected_result)

    >>> x = torch.from_numpy(np.random.rand(10))
    >>> y = torch.from_numpy(np.random.rand(3))
    >>> result = correlate_signals(x=x, y=y, fft_size=len(x), out_size=(len(x)-len(y) + 1))
    >>> expected_result = np.correlate(x, y, mode='valid')
    >>> np.testing.assert_array_almost_equal(result, expected_result)

    >>> x = torch.from_numpy(np.random.rand(100))
    >>> y = torch.from_numpy(np.random.rand(11))
    >>> result = correlate_signals(x=x, y=y, fft_size=len(x), out_size=(len(x)-len(y) + 1))
    >>> expected_result = np.correlate(x, y, mode='valid')
    >>> np.testing.assert_array_almost_equal(result, expected_result)

    >>> x = tensor([[[1.0,2.0,3.0,4.0]]])
    >>> y = tensor([[[1.0,3.0]]])
    >>> s = x.shape[-1]- y.shape[-1] + 1
    >>> result = correlate_signals(x=x, y=y, fft_size=x.shape[-1], out_size=s)
    >>> expect = np.array([[[7.0, 11.0, 15.0]]])
    >>> np.testing.assert_array_almost_equal(result, expect)
    """
    # pad the signals to the fft size
    x = F.pad(x, (0, fft_size - x.shape[-1]), 'constant', 0.0)
    y = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    # onesided=True: only the frequency coefficients to the Nyquist
    # frequency are retained (about half the length of the
    # input signal) so the original signal can be still exactly
    # reconstructed from the frequency samples.
    xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=True)
    yfft = torch.rfft(y, signal_ndim=signal_ndim, onesided=True)
    if preserve_energy_rate is not None or index_back is not None:
        index_xfft = preserve_energy_index(xfft, preserve_energy_rate,
                                           index_back)
        index_yfft = preserve_energy_index(yfft, preserve_energy_rate,
                                           index_back)
        index = max(index_xfft, index_yfft)
        # with open(log_file, "a+") as f:
        #     f.write("index: " + str(compress_rate) +
        # ";preserved energy input: " + str(
        #         compute_energy(xfft[:index]) / compute_energy(
        # xfft[:fft_size // 2 + 1])) +
        #             ";preserved energy filter: " + str(
        #         compute_energy(yfft[:index]) / compute_energy(
        # yfft[:fft_size // 2 + 1])) + "\n")

        # complex numbers are represented as the pair of numbers in
        # the last dimension so we have to narrow the length
        # of the last but one dimension
        xfft = xfft.narrow(dim=-2, start=0, length=index)
        yfft = yfft.narrow(dim=-2, start=0, length=index)
        # print("the signal size after compression: ", index)

        input = complex_mul(xfft, pytorch_conjugate(yfft))
        # we need to pad complex numbers expressed as a pair of real
        # numbers in the last dimension
        # xfft = F.pad(input=xfft, pad=(0, fft_size - index),
        # mode='constant', value=0)
        # yfft = F.pad(input=yfft, pad=(0, fft_size - index),
        # mode='constant', value=0)
        pad_shape = tensor(xfft.shape)
        # xfft has at least two dimension (with the last one being a
        # dimension for a pair of real number representing a complex
        # number.
        pad_shape[-2] = (fft_size // 2 + 1) - index
        complex_pad = torch.zeros(*pad_shape, dtype=xfft.dtype,
                                  device=xfft.device)
        input = torch.cat((input, complex_pad), dim=-2)
    else:
        input = complex_mul(xfft, pytorch_conjugate(yfft))
    out = torch.irfft(input, signal_ndim=signal_ndim,
                      signal_sizes=(x.shape[-1],))

    # plot_signal(out, "out after ifft")
    out = out[..., :out_size]
    # plot_signal(out, "after truncating to xlen: " + str(x_len))

    # sum the values across the computed layers (for each filter)
    # out = torch.sum(out, dim=-2, keepdim=True)
    return out


def correlate_fft_signals(xfft, yfft, fft_size: int,
                          signal_ndim: int = 1) -> object:
    """
    Similar to 'correlate_signal' function but the signals are provided in the
    frequency domain (after fft) for the reuse of the maps.

    Cross-correlation of the signals: x and y.
    Theory: X(f) = fft(x(t)). The first sample X(0) of the
    transformed series is the DC component, more commonly known
    as the average of the input series. For the normalized fft
    (both sums are multiplied by $\frac{1}{\sqrt{N}}$. The length of the sin(x)
    is || sin(x)||^2 = \integral_{0}^{2\pi} sin^2(x) dx = \pi, so
    ||sin(x)||=\sqrt(\pi)

    $$
    \begin{align}
        X(0) = \frac{1}{\sqrt{N}} \sum_{n=0}^{n=N-1} x(n)
    \end{align}
    $$

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # complex_mul_cpp only broadcasts the input if we provide all filters
    # but staying on this level is probably more performant than the other
    # approach (with full broadcast of the input and output) since we use less
    # memory.
    # >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,3.0,3.0,2.0]],
    # ... [[2.0,1.0,3.0,5.0], [1.0,2.0,5.0,1.0]]])
    # >>> # two filters
    # >>> y = tensor([[[1.0,3.0], [0.0,3.0]], [[1.0,1.0], [2.0,2.0]]])
    # >>> fft_size = x.shape[-1]
    # >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    # >>> signal_ndim = 1
    # >>> onesided = True
    # >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    # >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    # >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft,
    # ... fft_size=x.shape[-1], out_size=(x.shape[-1]-y.shape[-1] + 1),
    # ... input_size=x.shape[-1])
    # >>> # print("result: ", result)
    # >>> np.testing.assert_array_almost_equal(result,
    # ... np.array([[[16.0, 20.0, 21.0], [11.0, 17.0, 17.0]],
    # ... [[11.0, 25.0, 21.0], [9.0, 18.0, 20.0]]]))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    >>> # Test the backward computation without summing up the final tensor.
    >>> # The summing up of the final tensor is done only if param is_forward
    >>> # is set to True.
    >>> x = tensor([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
    >>> # Two filters.
    >>> y = tensor([[[0.1, -0.2]]])
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft,
    ... fft_size=x.shape[-1])
    >>> # print("result: ", result)
    >>> out_size=(x.shape[-1]-y.shape[-1] + 1)
    >>> np.testing.assert_array_almost_equal(result[...,:out_size],
    ... np.array([[[-0.3, -0.4], [ 0.5, -0.7]]]))

    >>> x = tensor([[[2.0,1.0,3.0,5.0], [1.0,2.0,5.0,1.0]]])
    >>> # two filters
    >>> y = tensor([[[1.0,3.0], [0.0,3.0]], [[1.0,1.0], [2.0,2.0]]])
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft,
    ... fft_size=x.shape[-1])
    >>> # print("result: ", result)
    >>> out_size=(x.shape[-1]-y.shape[-1] + 1)
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... np.array([[[ 5., 10., 18.], [ 6., 15.,  3.]],
    ... [[ 3.,  4.,  8.], [ 6., 14., 12.]]]))

    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,3.0,3.0,2.0]]])
    >>> # two filters
    >>> y = tensor([[[1.0,3.0], [0.0,3.0]], [[1.0,1.0], [2.0,2.0]]])
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft,
    ... fft_size=x.shape[-1])
    >>> # print("result: ", result)
    >>> out_size=(x.shape[-1]-y.shape[-1] + 1)
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... np.array([[[ 7., 11., 15.], [ 9.,  9.,  6.]],
    ... [[ 3.,  5.,  7.], [ 8., 12., 10.]]]))

    >>> # 1 signal in the batch, 2 channels, signal of length 3
    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,2.0,3.0,4.0]]])
    >>> y = tensor([[[1.0,3.0], [1.0,3.0]]])
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft,
    ... fft_size=x.shape[-1])
    >>> # print("result: ", result)
    >>> out_size=(x.shape[-1]-y.shape[-1] + 1)
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... np.array([[[ 7., 11., 15.], [ 7., 11., 15.]]]))

    >>> x = tensor([1.0,2.0,3.0,4.0])
    >>> y = tensor([1.0,3.0])
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft, fft_size=len(x))
    >>> out_size=len(x)-len(y) + 1
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... np.array([7.0, 11.0, 15.0]))

    >>> x = torch.from_numpy(np.random.rand(10))
    >>> y = torch.from_numpy(np.random.rand(3))
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> # print("y_padded: ", y_padded)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft, fft_size=len(x))
    >>> out_size=len(x)-len(y) + 1
    >>> expected_result = np.correlate(x, y, mode='valid')
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... expected_result)

    >>> x = torch.from_numpy(np.random.rand(100))
    >>> y = torch.from_numpy(np.random.rand(11))
    >>> fft_size = x.shape[-1]
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> # print("y_padded: ", y_padded)
    >>> signal_ndim = 1
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals(xfft=xfft, yfft=yfft, fft_size=len(x))
    >>> out_size=x.shape[-1]-y.shape[-1] + 1
    >>> expected_result = np.correlate(x, y, mode='valid')
    >>> np.testing.assert_array_almost_equal(result[..., :out_size],
    ... expected_result)

    :param xfft: input signal after fft
    :param yfft: filter after fft
    :param fft_size: the size of the signal in the frequency domain
    :param signal_ndim: the dimension of the signal (we set it to 1)
    :return: output signal after correlation of signals xfft and yfft
    """
    # xfft = complex_pad_simple(xfft=xfft, fft_size=fft_size)
    # yfft = complex_pad_simple(xfft=yfft, fft_size=fft_size)
    freq_mul = complex_mul(xfft, pytorch_conjugate(yfft))
    freq_mul = complex_pad_simple(xfft=freq_mul, fft_size=fft_size)
    # print("freq mul size: ", freq_mul.size())
    out = torch.irfft(
        input=freq_mul, signal_ndim=signal_ndim, signal_sizes=(fft_size,))
    del freq_mul
    return out


def correlate_dct_signals(xdct, ydct, dct_size):
    """
    :param xdct: input dct signal
    :param ydct: input dct filter
    :return: the correlation between xdct and ydct
    
    >>> # Test the backward computation without summing up the final tensor.
    >>> # The summing up of the final tensor is done only if param is_forward
    >>> # is set to True.
    >>> x = tensor([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
    >>> # Two filters.
    >>> y = tensor([[[0.1, -0.2]]])
    >>> fft_size = x.shape[-1] + y.shape[-1]//2
    >>> y_padded = F.pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
    >>> signal_ndim = 1
    >>> xdct = torch_dct.dct(x)
    >>> ydct = torch_dct.dct(y)
    >>> result = correlate_dct_signals(xfft=xfft, yfft=yfft,
    ... fft_size=fft_size)
    >>> # print("result: ", result)
    >>> out_size=(x.shape[-1]-y.shape[-1] + 1)
    >>> np.testing.assert_array_almost_equal(result[...,:out_size],
    ... np.array([[[-0.3, -0.4], [ 0.5, -0.7]]]))
    """
    freq_mul = xdct * pytorch_conjugate(ydct)
    freq_mul = F.pad(freq_mul, (0, dct_size - freq_mul.shape[-1]),
                     mode='constant')
    # print("freq mul size: ", freq_mul.size())
    out = torch_dct.idct(freq_mul)
    del freq_mul
    return out


def correlate_dct_1D(x, y, use_next_power2=True):
    """
    Correlate 1D input signals via the DCT transformation.

    :param x: input signal in the time domain.
    :param y: input filter in the time domain.
    :return: the correlation z between x and y
    """
    N = x.shape[-1]
    L = y.shape[-1]
    M = N + L - 1
    P1 = max((L - 3), 0) // 2 + 1
    P2 = max(P1 + 1, (N - 3) // 2 + 1)
    P = max(P2 + 1, 3 * M // 2 + 1)

    if use_next_power2:
        P = next_power2(P)

    x = F.pad(input=x, pad=(P1, P - P1 - N), mode='constant')
    y = y.flip([-1])
    y = F.pad(input=y, pad=(P2, P - P2 - L), mode='constant')
    x = torch_dct.dct(x)
    y = torch_dct.dct(y)
    z = x * y
    z = torch_dct.dct1(z)
    return z


def correlate_fft_signals2D(xfft, yfft, input_height, input_width,
                            init_fft_height, init_half_fft_width,
                            out_height, out_width, is_forward=True):
    """
    >>> # Test 2 channels and 2 filters.
    >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
    ... [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]])
    >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0],[3.0, -2.0]]],
    ... [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]])
    >>> fft_width = x.shape[-1]
    >>> fft_height = x.shape[-2]
    >>> pad_right = fft_width - y.shape[-1]
    >>> pad_bottom = fft_height - y.shape[-2]
    >>> y_padded = F.pad(y, (0, pad_right, 0, pad_bottom), 'constant', 0.0)
    >>> np.testing.assert_array_equal(x=tensor([
    ... [[[1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
    ... [[-1.0, 2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 0.0, 0.0]]],
    ... [[[-1.0, 1.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
    ... [[-2.0, 1.0, 0.0], [1.0, -3.0, 0.0], [0.0, 0.0, 0.0]]]]), y=y_padded,
    ... err_msg="The expected result x is different than the computed y.")
    >>> signal_ndim = 2
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals2D(xfft=xfft,
    ... yfft=pytorch_conjugate(yfft),
    ... input_height=fft_height, input_width=fft_width,
    ... init_fft_height=xfft.shape[-3], init_half_fft_width=xfft.shape[-2],
    ... out_height=(x.shape[-2]-y.shape[-2] + 1),
    ... out_width=(x.shape[-1]-y.shape[-1] + 1))
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(
    ... x=np.array([[[[23.0, 32.0], [30., 4.]],[[11.0, 12.0], [13.0, -11.0]]]]),
    ... y=result, decimal=5,
    ... err_msg="The expected array x and computed y are not almost equal.")

    >>> # Test 2D convolution.
    >>> x = tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]])
    >>> # A single filter.
    >>> y = tensor([[[1.0, 2.0], [3.0, 2.0]]])
    >>> fft_width = x.shape[-1]
    >>> fft_height = x.shape[-2]
    >>> pad_right = fft_width - y.shape[-1]
    >>> pad_bottom = fft_height - y.shape[-2]
    >>> y_padded = F.pad(y, (0, pad_right, 0, pad_bottom), 'constant', 0.0)
    >>> np.testing.assert_array_equal(x=tensor([[[1.0, 2.0, 0.0],
    ... [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]]]), y=y_padded,
    ... err_msg="The expected result x is different than the computed y.")
    >>> signal_ndim = 2
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals2D(xfft=xfft, yfft=pytorch_conjugate(yfft),
    ... input_height=fft_height, input_width=fft_width,
    ... init_fft_height=xfft.shape[-2], init_half_fft_width=xfft.shape[-1],
    ... out_height=(x.shape[-2]-y.shape[-2]+1),
    ... out_width=(x.shape[-1]-y.shape[-1] + 1))
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(
    ... x=np.array([[[22.0, 22.0], [18., 14.]]]), y=result,
    ... err_msg="The expected array x and computed y are not almost equal.")

    >>> # Test 2D convolution.
    >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
    >>> # A single filter.
    >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
    >>> fft_width = x.shape[-1]
    >>> fft_height = x.shape[-2]
    >>> pad_right = fft_width - y.shape[-1]
    >>> pad_bottom = fft_height - y.shape[-2]
    >>> y_padded = F.pad(y, (0, pad_right, 0, pad_bottom), 'constant', 0.0)
    >>> np.testing.assert_array_equal(x=tensor([[[[1.0, 2.0, 0.0],
    ... [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]]]]), y=y_padded,
    ... err_msg="The expected result x is different than the computed y.")
    >>> signal_ndim = 2
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals2D(xfft=xfft, yfft=pytorch_conjugate(yfft),
    ... input_height=fft_height, input_width=fft_width,
    ... init_fft_height=xfft.shape[-3], init_half_fft_width=xfft.shape[-2],
    ... out_height=(x.shape[-2]-y.shape[-2]+1),
    ... out_width=(x.shape[-1]-y.shape[-1] + 1))
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(
    ... x=np.array([[[[22.0, 22.0], [18., 14.]]]]), y=result,
    ... err_msg="The expected array x and computed y are not almost equal.")

    >>> # Test 2 channels.
    >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
    ... [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]])
    >>> # A single filter.
    >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0],[3.0, -2.0]]]])
    >>> fft_width = x.shape[-1]
    >>> fft_height = x.shape[-2]
    >>> pad_right = fft_width - y.shape[-1]
    >>> pad_bottom = fft_height - y.shape[-2]
    >>> y_padded = F.pad(y, (0, pad_right, 0, pad_bottom), 'constant', 0.0)
    >>> np.testing.assert_array_equal(x=tensor([[
    ... [[1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
    ... [[-1.0, 2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 0.0, 0.0]]
    ... ]]), y=y_padded,
    ... err_msg="The expected result x is different than the computed y.")
    >>> signal_ndim = 2
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals2D(xfft=xfft, yfft=pytorch_conjugate(yfft),
    ... input_height=fft_height, input_width=fft_width,
    ... init_fft_height=xfft.shape[-3], init_half_fft_width=xfft.shape[-2],
    ... out_height=(x.shape[-2]-y.shape[-2]+1),
    ... out_width=(x.shape[-1]-y.shape[-1] + 1))
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(
    ... x=np.array([[[[23.0, 32.0], [30., 4.]]]]), y=result, decimal=5,
    ... err_msg="The expected array x and computed y are not almost equal.")

    :param xfft: first input map
    :param yfft: second input map
    :param input_height: the height of input x
    :param input_widt: the width of input x
    :param fft_height: the fft height for maps (both input maps xfft and yfft
    for cross-correlation have to have the same dimensions).
    :param fft_width: the fft width for maps (both input maps xfft and yfft
    for cross-correlation have to have the same dimensions).
    :param out_height: the height of the output map
    :param out_width: the width of the output map
    :param is_forward: is the correlation for a forward of a backward pass of
    the convolution operation.
    :return: output map after correlation of xfft with yfft
    """
    signal_ndim = 2

    # xfft = complex_pad2D(fft_input=xfft, half_fft_height=half_fft_height,
    #                      half_fft_width=half_fft_width)
    # yfft = complex_pad2D(fft_input=yfft, half_fft_height=half_fft_height,
    #                      half_fft_width=half_fft_width)
    freq_mul = complex_mul(xfft, yfft)
    if freq_mul.dim() == 6:
        # For bulk multiplication.
        freq_mul = restore_size_2D_batch(xfft=freq_mul,
                                         init_H_fft=init_fft_height,
                                         init_half_W_fft=init_half_fft_width)
    elif freq_mul.dim() == 5:
        # For serial multiplication.
        freq_mul = restore_size_2D(xfft=freq_mul,
                                   init_H_fft=init_fft_height,
                                   init_half_W_fft=init_half_fft_width)
    else:
        raise Exception(f"Unsupported number of dimensions: {xfft.dim()}")
    out = torch.irfft(input=freq_mul, signal_ndim=signal_ndim,
                      signal_sizes=(input_height, input_width), onesided=True)
    all_tensors_size = get_tensors_elem_size()
    # print("all tensor size in corr: ", all_tensors_size / 2 ** 30)
    # print("torch max memory in corr: ", torch.cuda.max_memory_allocated() / 2 ** 30)
    del freq_mul

    out = out[..., :out_height, :out_width]
    if out.dim() > 2 and is_forward:
        out = torch.sum(input=out, dim=1, keepdim=False)
    return out


def complex_pad(xfft, fft_size):
    """
    >>> # Typical use case.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]])
    >>> expected_xfft_pad = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0],
    ... [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    >>> half_fft_size = xfft.shape[-2] + 3
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad = complex_pad(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    >>> # Only two dimensions (the -2 being the "true one" and the last one is
    >>> # for the complex numbers.
    >>> xfft = tensor([[1.0, 0.0], [0.0, 4.0], [4.0, 5.0], [-1.0, 0.0]])
    >>> expected_xfft_pad = tensor([[1.0, 0.0], [0.0, 4.0], [4.0, 5.0],
    ... [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    >>> half_fft_size = xfft.shape[-2] + 2
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad = complex_pad(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    >>> # Check if it works for the case where padding should not be done.
    >>> # So expected result is the same as the xfft.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]])
    >>> half_fft_size = xfft.shape[-2]
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad = complex_pad(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=xfft, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    Pad xfft with zeros in the frequency domain to the size specified with
    fft_size.

    :param xfft: the input signal in the frequency domain (represented by
    complex numbers).
    :param fft_size: the expected (initial) fft size (used in the forward pass).
    :return: the padded xfft signal with zeros in the frequency domain.
    """
    # xfft has at least two dimensions (with the last one being a dimension for
    # a pair of real numbers representing a complex number). Moreover, pytorch
    # supports half-sized fft (one-sided fft) by default.
    half_fft = fft_size // 2 + 1
    pad_shape = tensor(xfft.shape)
    # Omit the last dimension (-1) for complex numbers.
    current_length = xfft.shape[-2]
    if current_length < half_fft:
        pad_shape[-2] = half_fft - current_length
        complex_pad = torch.zeros(*pad_shape, dtype=xfft.dtype,
                                  device=xfft.device)
        xfft = torch.cat((xfft, complex_pad), dim=-2)
    return xfft


def complex_pad_simple(xfft, fft_size):
    """
    >>> # Typical use case.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]])
    >>> expected_xfft_pad = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0],
    ... [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    >>> half_fft_size = xfft.shape[-2] + 3
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad_simple = complex_pad_simple(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad,
    ... y=xfft_pad_simple,
    ... err_msg="The expected x is different than computed y")

    >>> # Only two dimensions (the -2 being the "true one" and the last one is
    >>> # for the complex numbers.
    >>> xfft = tensor([[1.0, 0.0], [0.0, 4.0], [4.0, 5.0], [-1.0, 0.0]])
    >>> expected_xfft_pad = tensor([[1.0, 0.0], [0.0, 4.0], [4.0, 5.0],
    ... [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    >>> half_fft_size = xfft.shape[-2] + 2
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad = complex_pad_simple(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    >>> # Check if it works for the case where padding should not be done.
    >>> # So expected result is the same as the xfft.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]])
    >>> half_fft_size = xfft.shape[-2]
    >>> fft_size = (half_fft_size - 1) * 2
    >>> xfft_pad = complex_pad_simple(xfft, fft_size)
    >>> np.testing.assert_array_almost_equal(x=xfft, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    Use the torch.nn.functional.pad.

    :param xfft: the fft-ed signal (containing complex numbers)
    :param fft_size: the initial size of the fft (before oneside-ing it).
    :return: the padded xfft signal.
    """
    half_fft = fft_size // 2 + 1
    current_length = xfft.shape[-2]
    if half_fft > current_length:
        pad_right = half_fft - current_length
        # We have to skip the last dimension that represents the complex number
        # so effectively we use the 2D padding for the 1D complex values.
        return F.pad(input=xfft, pad=(0, 0, 0, pad_right), mode="constant",
                     value=0)
    else:
        return xfft


def complex_pad2D(fft_input, half_fft_height, half_fft_width):
    """
    >>> # Typical use case.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]],
    ... [[2.0, 1.0], [-1.0, 2.0], [5.0, 1.0]]])
    >>> expected_xfft_pad = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0],
    ... [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ... [[2.0, 1.0], [-1.0, 2.0], [5.0, 1.0], [0., 0.], [0., 0.], [0., 0.]],
    ... [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0., 0.], [0., 0.], [0., 0.]]])
    >>> half_fft_width = xfft.shape[-2] + 3  # width
    >>> half_fft_height = xfft.shape[-3] + 1  # height
    >>> xfft_pad = complex_pad2D(fft_input=xfft, half_fft_height=half_fft_height,
    ... half_fft_width=half_fft_width)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    >>> # Only two dimensions (the -2 being the "true one" and the last one is
    >>> # for the complex numbers.
    >>> xfft = tensor([[[1.0, 0.0], [0.0, 4.0], [4.0, 5.0], [-1.0, 0.0]]])
    >>> expected_xfft_pad = tensor([[[1.0, 0.0], [0.0, 4.0], [4.0, 5.0],
    ... [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0],
    ... [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    >>> half_fft_width = xfft.shape[-2] + 2
    >>> half_fft_height = xfft.shape[-3] + 1
    >>> xfft_pad = complex_pad2D(fft_input=xfft, half_fft_height=half_fft_height,
    ... half_fft_width=half_fft_width)
    >>> np.testing.assert_array_almost_equal(x=expected_xfft_pad, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    >>> # Check if works for the case where padding should not be done.
    >>> # So expected result is the same as the xfft.
    >>> xfft = tensor([[[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]])
    >>> half_fft_width = xfft.shape[-2]
    >>> half_fft_height = xfft.shape[-3]
    >>> xfft_pad = complex_pad2D(fft_input=xfft, half_fft_height=half_fft_height,
    ... half_fft_width=half_fft_width)
    >>> np.testing.assert_array_almost_equal(x=xfft, y=xfft_pad,
    ... err_msg="The expected x is different than computed y")

    Pad xfft with zeros in the frequency domain to the size specified with
    fft_height and fft_width.

    :param xfft: the input signal in the frequency domain (represented by
    complex numbers).
    :param half_fft_height: the expected initial half fft height (the last but
    one dimension for real numbers and one further "into depth" dimension for
    complex numbers) used in the forward pass.
    :param half_fft_width: the expected initial half fft width (the last
    dimension for real numbers and the one more "into depth" dimension for
    complex numbers) used in the forward pass.
    :return: the padded xfft signal with zeros in the frequency domain.
    """
    # Omit the last dimension (-1) for complex numbers.
    current_height = fft_input.shape[-3]
    current_width = fft_input.shape[-2]
    if current_height < half_fft_height or current_width < half_fft_width:
        pad_bottom = half_fft_height - current_height
        pad_right = half_fft_width - current_width
        return F.pad(fft_input, (0, 0, 0, pad_right, 0, pad_bottom))
    return fft_input


def fast_jmul(input, filter):
    """
    Fast complex multiplication.

    :param input: fft-ed complex input.
    :param filter: fft-ed complex.

    input_re: real part of the input (a).
    input_im: imaginary part of the input (b).
    filter_re: real part of the filter (c).
    filter_im: imaginary part of the filter (d).

    :return: the result of the complex multiplication.

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> # complex multiply for 2D (complex) tensors
    >>> x = tensor([[[[1., 2.], [5., 5.]], [[2., 1.], [3., 3.]]]])
    >>> y = tensor([[[[2., 3.], [-1., 2.]], [[0.0, 2.0], [2., 1.]]]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[[-4., 7.], [-15., 5.0]],
    ... [[-2., 4.0], [3.0, 9.]]]]))

    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(fast_jmul(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(fast_jmul(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))

    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))

    >>> x = tensor([[5., 5.]])
    >>> y = tensor([[-1., 2.]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-15., 5.]]))

    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))

    >>> x = tensor([[[1., 2.], [3., 1.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]]]))

    >>> x = tensor([[[1., 2.], [3., 1.]], [[ 6.,  0.], [-2., -2.]]])
    >>> y = tensor([[[2., 3.], [1., 2.]], [[18.,  0.], [-2., -6.]]])
    >>> xy = fast_jmul(x, y)
    >>> np.testing.assert_array_equal(xy,
    ... tensor([[[-4., 7.], [1., 7.]], [[108.,   0.], [ -8.,  16.]]]))
    >>> del x
    >>> del y
    >>> del xy
    """
    a = input[..., :1]
    b = input[..., 1:]
    c = filter[..., :1]
    d = filter[..., 1:]

    ac = torch.mul(a, c)
    bd = torch.mul(b, d)

    a_bc_d = torch.mul(torch.add(a, b), torch.add(c, d))
    out_re = ac - bd
    out_im = a_bc_d - ac - bd

    out = torch.cat((out_re, out_im), -1)

    return out


def retain_low_coef(xfft, preserve_energy=None, index_back=None):
    """
    Retain the low coefficients to either to reach the required
    preserve_energy or after removing compress_rate coefficients. Only one of them
    should be chosen. The coefficients with the highest frequencies are
    discarded (they usually represent noise for naturla signals and images).

    :param xfft: the input signal (4 dimensions: batch size, channel, signal,
    complex numbers).
    :param preserve_energy: the percentage of energy to be preserved
    :param index_back: the number of zeroed out coefficients (starting from the
    smallest one).
    :return: the zeroed-out small coefficients

    >>> # Simple compress_rate.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_low_coef(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_low_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[0.1, 0.1], [30., 40.], [1.1, 2.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_low_coef(xfft, preserve_energy=5)
    >>> expected = torch.tensor([[[[0.1, 0.1], [30., 40.], [0.0, 0.0],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple index back.
    >>> xfft = torch.tensor([[[[1.1, 2.1], [30., 40.], [0.1, 0.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_low_coef(xfft, compress_rate=3)
    >>> expected = torch.tensor([[[[1.1, 2.1], [30., 40.], [0.0, 0.0],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_low_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_low_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_low_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_low_coef(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())
    >>> del xfft
    >>> del expected
    """
    INPUT_ERROR = "Specify only one of: compress_rate, preserve_energy"
    if (index_back is not None and index_back > 0) and (
            preserve_energy is not None and preserve_energy < 100):
        raise TypeError(INPUT_ERROR)
    if xfft is None or len(xfft) == 0:
        return xfft
    if (preserve_energy is not None and preserve_energy < 100) or (
            index_back is not None and index_back > 0):
        out = torch.zeros_like(xfft, device=xfft.device)
        for data_point_index, data_point_value in enumerate(xfft):
            for channel_index, channel_value in enumerate(data_point_value):
                full_energy, squared = get_full_energy(channel_value)
                current_energy = 0.0
                preserved_indexes = len(squared)
                if index_back is not None:
                    preserved_indexes = len(squared) - index_back
                preserved_energy = full_energy
                if preserve_energy is not None:
                    preserved_energy = full_energy * preserve_energy / 100
                index = 0
                while current_energy < preserved_energy and (
                        index < preserved_indexes):
                    energy = squared[index]
                    # np.testing.assert_almost_equal(actual=energy,
                    #                                desired=squared[coeff_index])
                    current_energy += energy
                    out[data_point_index, channel_index, index, :] = \
                        xfft[data_point_index, channel_index, index, :]
                    index += 1
        return out
    return xfft


def retain_big_coef(xfft, preserve_energy=None, index_back=None):
    """
    Retain the largest coefficients to either to reach the required
    preserve_energy or after removing compress_rate coefficients. Only one of them
    should be chosen.

    :param xfft: the input signal (4 dimensions: batch size, channel, signal,
    complex numbers).
    :param preserve_energy: the percentage of energy to be preserved
    :param index_back: the number of zeroed out coefficients (starting from the
    smallest one).
    :return: the zeroed-out small coefficients

    >>> # Simple compress_rate.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_big_coef(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_big_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[1.1, 2.1], [30., 40.], [0.1, 0.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_big_coef(xfft, preserve_energy=5)
    >>> expected = torch.tensor([[[[0.0, 0.0], [30., 40.], [0.0, 0.0],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple index back.
    >>> xfft = torch.tensor([[[[0.1, 0.1], [30., 40.], [1.1, 2.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_big_coef(xfft, compress_rate=3)
    >>> expected = torch.tensor([[[[0.0, 0.0], [30., 40.], [1.1, 2.1],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.01, 0.002], [2.0, -6.0], [0.0, 0.1]]]])
    >>> result = retain_big_coef(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.0], [2.0, -6.0], [0.0, 0.1]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())
    >>> del xfft
    >>> del expected
    """
    INPUT_ERROR = "Specify only one of: compress_rate, preserve_energy"
    if (index_back is not None and index_back > 0) and (
            preserve_energy is not None and preserve_energy < 100):
        raise TypeError(INPUT_ERROR)
    if xfft is None or len(xfft) == 0:
        return xfft
    if (preserve_energy is not None and preserve_energy < 100) or (
            index_back is not None and index_back > 0):
        out = torch.zeros_like(xfft, device=xfft.device)
        for data_point_index, data_point_value in enumerate(xfft):
            for channel_index, channel_value in enumerate(data_point_value):
                full_energy, squared = get_full_energy(channel_value)
                current_energy = 0.0
                added_indexes = 0
                preserved_indexes = len(squared)
                if index_back is not None:
                    preserved_indexes = len(squared) - index_back
                preserved_energy = full_energy
                if preserve_energy is not None:
                    preserved_energy = full_energy * preserve_energy / 100
                # Creates the priority queue with the pair: coefficient
                # absolute value and the position (index) in the signal
                # (array).
                heap = []
                # Index and value of a coefficient.
                for square_index, square_value in enumerate(squared):
                    # We want to get the largest coefficients.
                    heappush(heap, (-square_value, square_index))
                while current_energy < preserved_energy and (
                        added_indexes < preserved_indexes):
                    neg_energy, coeff_index = heappop(heap)
                    energy = (-neg_energy.item())
                    # np.testing.assert_almost_equal(actual=energy,
                    #                                desired=squared[coeff_index])
                    current_energy += (-neg_energy)
                    added_indexes += 1
                    out[data_point_index, channel_index, coeff_index, :] = \
                        xfft[data_point_index, channel_index, coeff_index, :]
        return out
    return xfft


def retain_big_coef_bulk(xfft, preserve_energy=None, index_back=None):
    """
    Retain the largest coefficients to either to reach the required
    preserve_energy or after removing compress_rate coefficients. Only one of them
    should be chosen.

    :param xfft: the input signal (4 dimensions: batch size, channel, signal,
    complex numbers).
    :param preserve_energy: the percentage of energy to be preserved
    :param index_back: the number of zeroed out coefficients (starting from the
    smallest one).
    :return: the zeroed-out small coefficients

    >>> # Simple compress_rate.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_big_coef_bulk(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]]])
    >>> result = retain_big_coef_bulk(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple preserved energy.
    >>> xfft = torch.tensor([[[[1.1, 2.1], [30., 40.], [0.1, 0.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_big_coef_bulk(xfft, preserve_energy=5)
    >>> expected = torch.tensor([[[[0.0, 0.0], [30., 40.], [0.0, 0.0],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Simple index back.
    >>> xfft = torch.tensor([[[[0.1, 0.1], [30., 40.], [1.1, 2.1], [0.1, -0.8],
    ... [0.0, -1.0]]]])
    >>> result = retain_big_coef_bulk(xfft, compress_rate=3)
    >>> expected = torch.tensor([[[[0.0, 0.0], [30., 40.], [1.1, 2.1],
    ... [0.0, 0.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef_bulk(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for preserved energy.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef_bulk(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 channels for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]],
    ... [[0.0, 0.1], [2.0, -6.0], [0.01, 0.002]]]])
    >>> result = retain_big_coef_bulk(xfft, preserve_energy=90)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]],
    ... [[0.0, 0.0], [2.0, -6.0], [0.0, 0.0]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())

    >>> # Check 2 data points for index back.
    >>> xfft = torch.tensor([[[[1., 2.], [3., 4.], [0.1, 0.1]]],
    ... [[[0.01, 0.002], [2.0, -6.0], [0.0, 0.1]]]])
    >>> result = retain_big_coef_bulk(xfft, compress_rate=1)
    >>> expected = torch.tensor([[[[1., 2.], [3., 4.], [0.0, 0.0]]],
    ... [[[0.0, 0.0], [2.0, -6.0], [0.0, 0.1]]]])
    >>> np.testing.assert_equal(actual=result.numpy(), desired=expected.numpy())
    >>> del xfft
    >>> del expected
    """
    INPUT_ERROR = "Specify only one of: compress_rate, preserve_energy"
    if (index_back is not None and index_back > 0) and (
            preserve_energy is not None and preserve_energy < 100):
        raise TypeError(INPUT_ERROR)
    if xfft is None or len(xfft) == 0:
        return xfft
    if (preserve_energy is not None and preserve_energy < 100) or (
            index_back is not None and index_back > 0):
        out = torch.zeros_like(xfft, device=xfft.device)
        full_energy_bulk, squared_bulk = get_full_energy_bulk(xfft)
        # plot_signal_freq(squared_bulk[0, 0].numpy())
        squared_bulk, indices_bulk = torch.sort(squared_bulk, descending=True)
        N, C, _, _ = xfft.size()
        for data_point_index in range(N):
            for channel_index in range(C):
                full_energy = full_energy_bulk[data_point_index, channel_index]
                squared = squared_bulk[data_point_index, channel_index]
                indices = indices_bulk[data_point_index, channel_index]
                current_energy = 0.0
                added_index = 0
                # Number of coefficients in the squared array (the amplitudes of
                # the coefficients).
                preserved_indexes = len(squared)
                if index_back is not None:
                    preserved_indexes -= index_back
                preserved_energy = full_energy
                if preserve_energy is not None:
                    preserved_energy = full_energy * preserve_energy / 100
                while current_energy < preserved_energy and (
                        added_index < preserved_indexes):
                    energy = squared[added_index]
                    coeff_index = indices[added_index]
                    current_energy += energy
                    added_index += 1
                    out[data_point_index, channel_index, coeff_index, :] = \
                        xfft[data_point_index, channel_index, coeff_index, :]
        return out
    return xfft


def cuda_mem_show(is_debug=True, info="", omit_objs=[]):
    if torch.cuda.is_available() and is_debug is True:
        cuda_mem_empty(is_debug=is_debug)
        only_cuda = True
        tensor_size = get_tensors_elem_size(only_cuda=only_cuda,
                                            omit_objs=omit_objs)
        tensor_count = get_tensors_elem_count(only_cuda=only_cuda)
        with open(mem_log_file, "a") as f:
            f.write(
                f"info,{info},memory allocated,{torch.cuda.memory_allocated()},max memory allocated,{torch.cuda.max_memory_allocated()},memory cached,{torch.cuda.memory_cached()},max memory cached,{torch.cuda.max_memory_cached()},total nr (count) of cuda tensor elements,{tensor_count},total size of cuda tensors,{tensor_size}\n")


def cuda_mem_empty(is_debug=True):
    if torch.cuda.is_available() and is_debug is True:
        # print("cuda empty cache")
        torch.cuda.empty_cache()


def get_all_tensor_names():
    """
    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> t1 = tensor([1])
    >>> a3 = tensor([1, 2, 3])
    >>> # print(get_all_tensor_names())
    >>> names = get_all_tensor_names()
    >>> # print("tensor names: ", names)
    >>> # print("lenght of tensor names: ", len(names))
    >>> assert len(names) == 2
    >>> assert names == {'a3', 't1'}
    >>> del t1
    >>> del a3
    """
    tensor_names = set()
    try:
        for obj in gc.get_objects():
            obj_names = re.findall(r"'(\w+)': tensor\(", str(obj))
            tensor_names.update(obj_names)
    except:
        pass
    return tensor_names


def del_object(obj):
    del obj


def get_tensors(only_cuda=False, omit_objs=[]):
    """
    https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3?u=adam_dziedzic
    https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/19?u=adam_dziedzic

    :return: list of active PyTorch tensors
    >>> import torch
    >>> from torch import tensor
    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1], device=device)
    >>> a3 = tensor([[1, 2], [3, 4]], device=device)
    >>> # print(get_all_tensor_names())
    >>> tensors = [tensor_obj for tensor_obj in get_tensors(only_cuda=only_cuda)]
    >>> # print(tensors)
    >>> # We doubled each t1, a3 tensors because of the tensors collection.
    >>> expected_tensor_length = 2
    >>> assert len(tensors) == expected_tensor_length, f"Expected length of tensors {expected_tensor_length}, but got {len(tensors)}, the tensors: {tensors}"
    >>> exp_size = (2,2)
    >>> act_size = tensors[1].size()
    >>> assert exp_size == act_size, f"Expected size {exp_size} but got: {act_size}"
    >>> del t1
    >>> del a3
    >>> clean_gc_return = map((lambda obj: del_object(obj)), tensors)
    """
    add_all_tensors = False if only_cuda is True else True
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensors = {}

    # omit_obj_ids = [id(obj) for obj in omit_objs]

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensors[id(tensor)] = tensor

    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, 'saved_tensors') and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception as ex:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensors.values()  # return a list of tensors


def get_tensors_elem_count(only_cuda=True):
    """
    Get total number of elements in tensors.

    :return: the total number of elements (floats, doubles, etc.) in tensors.
    """
    tensors = get_tensors(only_cuda=only_cuda)
    total_elem = 0
    for tensor_obj in tensors:
        # sizes = [size for size in tensor_obj.size()]
        # if len(sizes) == 1:
        #     total_elem += sizes[0]
        # elif len(sizes) > 1:
        #     total_elem += reduce((lambda x, y: x * y), sizes)
        total_elem += tensor_obj.numel()
    return total_elem


def get_elem_size(tensor_obj):
    """

    :param tensor_obj: a tensor
    :return: the size in bytes of a single element in the tensor based on its
    type

    >>> t1 = tensor([1,2,3], dtype=torch.int32)
    >>> size = get_elem_size(t1)
    >>> assert size == 4
    >>> del t1
    >>> t2 = tensor([1.,2.,3.], dtype=torch.float)
    >>> size = get_elem_size(t2)
    >>> assert size == 4
    >>> del t2
    >>> t3 = tensor([1.,2.,3.], dtype=torch.float16)
    >>> size = get_elem_size(t3)
    >>> assert size == 2
    >>> del t3
    >>> t4 = tensor([1.,2.,3.], dtype=torch.double)
    >>> size = get_elem_size(t4)
    >>> assert size == 8
    >>> del t4
    """
    obj_type = tensor_obj.dtype

    if (obj_type is torch.float64) or (obj_type is torch.double) or (
            obj_type is torch.int64) or (obj_type is torch.long):
        return 8
    elif (obj_type is torch.float32) or (obj_type is torch.float) or (
            obj_type is torch.int32) or (obj_type is torch.int):
        return 4
    elif (obj_type is torch.float16) or (obj_type is torch.half) or (
            obj_type is torch.int16) or (obj_type is torch.short):
        return 2
    elif (obj_type is torch.int8) or (obj_type is torch.uint8):
        return 1
    else:
        raise AttributeError(f"Unknown torch type: {obj_type}")


def get_tensors_elem_size_count(only_cuda=True):
    """
    Get total size of elements in tensors.

    :return: the total size in bytes and total count (number) of elements
    (floats, doubles, etc.) in tensors.

    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    >>> t4 = tensor([1., 2., 3.], dtype=torch.double, device=device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> size, count = get_tensors_elem_size_count(only_cuda=only_cuda)
    >>> # print("tensors: ", get_tensors())
    >>> assert size == 40, f"Expected size 40, but got {size}"
    >>> assert count == 7, f"Expected count 7, but got {count}"
    """
    tensors = [tensor_obj for tensor_obj in get_tensors(only_cuda=only_cuda)]
    total_size = 0
    total_count = 0
    for tensor_obj in tensors:
        # sizes = [size for size in tensor_obj.size()]
        # if len(sizes) == 1:
        #     total_count += sizes[0]
        # elif len(sizes) > 1:
        #     total_count += reduce((lambda x, y: x * y), sizes)
        count = tensor_obj.numel()
        total_size += count * get_elem_size(tensor_obj)
        total_count += count
    return total_size, total_count


def get_tensors_elem_size(only_cuda=True, omit_objs=[]):
    """
    Get total size of elements in tensors.

    :param only_cuda: count only tensors on gpu
    :param omit_obs: omit the objects (for example, we
    don't want to count object twice, it saved_tensors in the context in the
    backward pass and the retrieved objects (tensors) from the context.

    :return: the total size in bytes and total count (number) of elements
    (floats, doubles, etc.) in tensors.

    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    >>> t4 = tensor([1., 2., 3.], dtype=torch.double, device=device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> size = get_tensors_elem_size(only_cuda=only_cuda)
    >>> assert size == 40
    """
    total_size = 0
    for tensor_obj in get_tensors(only_cuda=only_cuda, omit_objs=omit_objs):
        total_size += tensor_obj.numel() * get_elem_size(tensor_obj)
    return total_size


def global_arg(input, is_min=True):
    """

    :param input: input tensor,
    :return: indices of global min or max value.

    >>> a = tensor([[ 0.2549,  0.1900, -0.1968, -1.0813], [ 0.0429,  0.2016, -0.3347, -0.8325], [ 0.4934, -0.0643, -0.0109,  2.2520], [ 0.1923,  0.7899,  0.9620, -1.3912]])
    >>> idxs = global_arg(a, is_min=True)
    >>> np.testing.assert_equal(idxs.numpy(), np.array([3,3]))
    >>> idxs = global_arg(a, is_min=False)  # this is global arg max
    >>> np.testing.assert_equal(idxs.numpy(), np.array([2,3]))

    >>> b = tensor([[[0.2405, 0.0971], [0.7107, 0.7426], [0.7884, 0.3021]], [[0.5443, 0.5506], [0.5323, 0.0937], [0.8568, 0.7572]]])
    >>> idxs = global_arg(b, is_min=True)
    >>> np.testing.assert_equal(idxs.numpy(), np.array([1, 1, 1]))
    >>> idxs = global_arg(b, is_min=False)
    >>> np.testing.assert_equal(idxs.numpy(), np.array([1, 2, 0]))

    >>> c = torch.rand(3, 16, 15)
    >>> idxs = global_arg(c, is_min=True)
    >>> # print(idxs)
    >>> np.testing.assert_equal(c[idxs[0],idxs[1],idxs[2]].numpy(), c.min().numpy())
    >>> idxs = global_arg(c, is_min=False)
    >>> # print(idxs)
    >>> np.testing.assert_equal(c[idxs[0],idxs[1],idxs[2]].numpy(), c.max().numpy())
    """
    if is_min:
        rawidx = input.argmin()
    else:
        rawidx = input.argmax()
    idx = collections.deque()
    # inverted list of dimensions
    for adim in reversed(list(input.size())):
        idx.appendleft(rawidx % adim)
        rawidx = rawidx / adim
    idx = torch.tensor(idx)
    return idx


def zero_out_min_simple(input):
    """
    Zero out the min value in the input.

    :param input: the input tensor with 4 dimensions.
    :return: input with zeroed out min value.

    >>> a = torch.tensor([[[[0.5435, 0.4313], [0.7498, 0.6879],[0.3355, 0.9008]],[[0.3906, 0.7883],[0.3841, 0.5293],[0.0139, 0.0468]]]])
    >>> b = zero_out_min_simple(a)
    >>> expected = torch.tensor([[[[0.5435, 0.4313], [0.7498, 0.6879],[0.3355, 0.9008]],[[0.3906, 0.7883],[0.3841, 0.5293],[0.0, 0.0468]]]])
    >>> np.testing.assert_equal(expected.numpy(), b.numpy())
    """
    assert len(input.size()) == 4
    idx = global_arg(input, is_min=True)
    input[idx[0], idx[1], idx[2], idx[3]] = 0
    return input


def zero_out_min(input, spectrum, max=None):
    """
    Zero out the spectrum and corresponding element in input for a min elem
    in spectrum.

    :param input: the input tensor with 4 dimensions.
    :return: input with zeroed out min value.

    >>> a = tensor([[[[[ 3.6487,  0.0000], [-0.3913,  0.0000]], [[-0.3621, -0.1744], [ 0.3639, -0.5432]],[[-0.3621,  0.1744],[ 0.3639,  0.5432]]],[[[ 2.1530,  0.0000],[-0.5759,  0.0000]],[[ 0.6919, -0.7384],[-0.3086,  0.0972]],[[ 0.6919,  0.7384],[-0.3086, -0.0972]]]]])
    >>> spectrum_a = tensor([[[[3.6487, 0.3913], [0.4019, 0.6538],[0.4019, 0.6538]],[[2.1530, 0.5759],[1.0119, 0.3235],[1.0119, 0.3235]]]])
    >>> b, spectrum_b = zero_out_min(a, spectrum_a)
    >>> spectrum_expected = tensor([[[[3.6487, 0.3913], [0.4019, 0.6538],[0.4019, 0.6538]],[[2.1530, 0.5759],[1.0119, 4.6487],[1.0119, 0.3235]]]])
    >>> # print("spectrum_expeted: ", spectrum_expected)
    >>> # print("spectrum_b: ", spectrum_b)
    >>> np.testing.assert_array_almost_equal(spectrum_expected.numpy(), spectrum_b.numpy())
    >>> a_expected = tensor([[[[[ 3.6487,  0.0000], [-0.3913,  0.0000]], [[-0.3621, -0.1744], [ 0.3639, -0.5432]],[[-0.3621,  0.1744],[ 0.3639,  0.5432]]],[[[ 2.1530,  0.0000],[-0.5759,  0.0000]],[[ 0.6919, -0.7384],[0.0, 0.0]],[[ 0.6919,  0.7384],[-0.3086, -0.0972]]]]])
    >>> np.testing.assert_array_almost_equal(a_expected.numpy(), b.numpy(), decimal=4)
    """
    assert len(input.size()) == 5
    assert len(spectrum.size()) == 4
    idx = global_arg(spectrum, is_min=True)
    if max is None:
        spectrum_max = spectrum.max()
        if spectrum_max < float("inf"):
            max = spectrum_max + 1.0
        else:
            max = float("inf")
    # print(spectrum[idx[0], idx[1], idx[2], idx[3]])
    spectrum[idx[0], idx[1], idx[2], idx[3]] = max
    input[idx[0], idx[1], idx[2], idx[3], 0] = 0.0
    input[idx[0], idx[1], idx[2], idx[3], 1] = 0.0
    # print(spectrum)
    return input, spectrum


def get_step_estimate(xfft, yfft, memory_size, scale=32):
    X = 32
    total_size = memory_size * 2 ** 30  # total mem size in GB
    if len(xfft.shape) == 5:  # 2D data
        N, C, H, W, I = xfft.size()
    elif len(xfft.shape) == 4:  # 1D data
        N, C, W, I = xfft.size()
        H = 1
    else:
        raise Exception(f"Unexpected number of data "
                        f"dimensions: {len(xfft.shape)}")
    F = yfft.shape[0]  # number of filters
    item_size = get_elem_size(xfft)
    corr_out_in_size = ((N * F * H * W) + (N + F) * C * H * W * I) * item_size
    no_x_intermediate_size = scale * 3 * F * C * H * W * I * item_size

    # 3 = corr result is complex (2) + after rfft it is real (1)
    X = (total_size - corr_out_in_size) / no_x_intermediate_size
    intermediate_size = X * no_x_intermediate_size
    computed_size = corr_out_in_size + intermediate_size
    print("computed size: ", computed_size / 2 ** 30)
    print("all tensor size before correlation: ",
          get_tensors_elem_size() / 2 ** 30)
    print("torch max memory before correlation: ",
          torch.cuda.max_memory_allocated() / 2 ** 30)
    print("X size: ", int(X))
    step = int(X)
    return step


if __name__ == "__main__":
    import sys
    import doctest

    clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        t1 = tensor([1], device=device)
        a3 = tensor([[1, -2], [3, -4]], device=device)
        tensor_obj = None

        tensors = get_tensors(only_cuda=True)
        for tensor_obj in tensors:
            print("tensor: ", tensor_obj)
        del tensor_obj
        tensors = get_tensors(only_cuda=True)
        for tensor_obj in tensors:
            print("tensor_obj: ", tensor_obj)
        del tensor_obj
        total_elem_nr = get_tensors_elem_count()
        print("total number of elements (items) in all tensors: ",
              total_elem_nr)
        expected = 5
        assert total_elem_nr == 5, f"Expected {expected} but got: {total_elem_nr}"
        del t1
        del a3
        del tensors

    # you can run it from the console: python pytorch_util.py -v
    sys.exit(doctest.testmod()[0])
