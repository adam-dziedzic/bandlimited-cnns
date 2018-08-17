"""
Pytorch utils.

The latest version of Pytorch (in the main branch 2018.06.30)
supports tensor flipping.
"""
import numpy as np
import torch
from torch import tensor
from torch.nn.functional import pad as torch_pad


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


def next_power2(x):
    """
    :param x: an integer number
    :return: the power of 2 which is the larger than x but the
    smallest possible

    >>> result = next_power2(5)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(1)
    >>> np.testing.assert_equal(result, 1)
    >>> result = next_power2(2)
    >>> np.testing.assert_equal(result, 2)
    >>> result = next_power2(7)
    >>> np.testing.assert_equal(result, 8)
    >>> result = next_power2(9)
    >>> np.testing.assert_equal(result, 16)
    """
    # return math.pow(2, math.ceil(math.log2(x)))
    return int(2 ** np.ceil(np.log2(x)))


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
    >>> x = tensor([[ 6.,  0.], [0., -2.], [1., 0.], [ 1.,  1.],
    ... [1., 2.]])
    >>> y = tensor([[2.,  0.], [0., -6.], [0., 1.], [ 1.,  1.],
    ... [2., 3.]])
    >>> np.testing.assert_array_equal(complex_mul(x, y),
    ... tensor([[12.,   0.], [-12., 0.], [0., 1.], [ 0.,   2.],
    ... [-4., 7.]]))
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors
    >>> # have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul(x, y),
    ... tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))
    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))
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
    mul = torch.mul
    add = torch.add
    cat = torch.cat

    ua = x.narrow(-1, 0, 1)
    ud = x.narrow(-1, 1, 1)
    va = y.narrow(-1, 0, 1)
    vb = y.narrow(-1, 1, 1)
    ub = add(ua, ud)
    uc = add(ud, mul(ua, -1))
    vc = add(va, vb)
    uavc = mul(ua, vc)
    # relational part of the complex number
    result_rel = add(uavc, mul(mul(ub, vb),
                               -1))
    # imaginary part of the complex number
    result_im = add(mul(uc, va),
                    uavc)
    result = cat(tensors=(result_rel, result_im),
                 dim=-1)  # use the last dimension
    return result


def pytorch_conjugate(x):
    """
    Conjugate all the complex numbers in tensor x in place.

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
    """
    x.narrow(-1, 1, 1).mul_(-1)
    return x


def get_full_energy(x):
    """
    Return the full energy of the signal. The energy E(xfft) of a
    sequence xfft is defined as the sum of energies
    (squares of the amplitude |x|) at every point of the sequence.

    see: http://www.cs.cmu.edu/~christos/PUBLICATIONS.OLDER/
    sigmod94.pdf (equation 7)

    :param x: an array of complex numbers
    :return: the full energy of signal x

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


def preserve_energy_index(xfft, energy_rate=None, index_back=None):
    """
    To which index should we preserve the xfft signal (and discard
    the remaining coefficients). This is based on the provided
    energy_rate, or if the energy_rate is not provided, then
    index_back is applied.

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
    >>> result = preserve_energy_index(xfft, index_back=1)
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
        while current_energy < preserved_energy and index < len(
                squared):
            current_energy += squared[index]
            index += 1
        return max(index, 1)
    elif index_back is not None:
        return len(xfft) - index_back
    return len(xfft)


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
    x = torch_pad(x, (0, fft_size - x.shape[-1]), 'constant', 0.0)
    y = torch_pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
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
        #     f.write("index: " + str(index_back) +
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
        # xfft = torch_pad(input=xfft, pad=(0, fft_size - index),
        # mode='constant', value=0)
        # yfft = torch_pad(input=yfft, pad=(0, fft_size - index),
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


def correlate_fft_signals(x, y, fft_size, out_size, preserve_energy_rate=None,
                          index_back=None, signal_ndim=1):
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

    :param x: input fft-ed signal
    :param y: fft-ed filter
    :param fft_size: the size of the signal in the frequency domain
    :param out_size: required output len (size)
    :param preserve_energy_rate: compressed to this energy rate
    :param index_back: how many coefficients to remove
    :param signal_ndim: what is the dimension of the input data
    :return: output signal after correlation of signals x and y

    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,2.0,3.0,4.0]]])
    >>> # two filters
    >>> y = tensor([[[1.0,3.0], [1.0,3.0]], [[1.0,3.0], [1.0,3.0]]])
    >>> result = correlate_signals(x=x, y=y, fft_size=x.shape[-1],
    ... out_size=(x.shape[-1]-y.shape[-1] + 1))
    >>> np.testing.assert_array_almost_equal(result,
    ... np.array([[[7.0, 11.0, 15.0], [7.0, 11.0, 15.0]]]))

    >>> x = tensor([[[1.0,2.0,3.0,4.0], [1.0,2.0,3.0,4.0]]])
    >>> y = tensor([[[1.0,3.0], [1.0,3.0]]])
    >>> result = correlate_signals(x=x, y=y, fft_size=x.shape[-1],
    ... out_size=(x.shape[-1]-y.shape[-1] + 1))
    >>> np.testing.assert_array_almost_equal(result,
    ... np.array([[[7.0, 11.0, 15.0], [7.0, 11.0, 15.0]]]))

    >>> x = tensor([1.0,2.0,3.0,4.0])
    >>> y = tensor([1.0,3.0])
    >>> result = correlate_signals(x=x, y=y, fft_size=len(x),
    ... out_size=(len(x)-len(y) + 1))
    >>> np.testing.assert_array_almost_equal(result,
    ... np.array([7.0, 11.0, 15.0]))

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
    """
    # pad the signals to the fft size
    x = torch_pad(x, (0, fft_size - x.shape[-1]), 'constant', 0.0)
    y = torch_pad(y, (0, fft_size - y.shape[-1]), 'constant', 0.0)
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
        #     f.write("index: " + str(index_back) +
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

        # we need to pad complex numbers expressed as a pair of real
        # numbers in the last dimension
        # xfft = torch_pad(input=xfft, pad=(0, fft_size - index),
        # mode='constant', value=0)
        # yfft = torch_pad(input=yfft, pad=(0, fft_size - index),
        # mode='constant', value=0)
        pad_shape = tensor(xfft.shape)
        # xfft has at least two dimension (with the last one being a
        # dimension for a pair of real number representing a complex
        # number.
        pad_shape[-2] = (fft_size // 2 + 1) - index
        complex_pad = torch.zeros(*pad_shape, dtype=xfft.dtype,
                                  device=xfft.device)
        xfft = torch.cat((xfft, complex_pad), dim=-2)
        yfft = torch.cat((yfft, complex_pad), dim=-2)
    out = torch.irfft(
        input=complex_mul(xfft, pytorch_conjugate(yfft)),
        signal_ndim=signal_ndim, signal_sizes=(x.shape[-1],))

    # plot_signal(out, "out after ifft")
    out = out[..., :out_size]
    # plot_signal(out, "after truncating to xlen: " + str(x_len))
    return out


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
