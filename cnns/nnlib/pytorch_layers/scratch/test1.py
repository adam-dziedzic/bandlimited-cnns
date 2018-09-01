from torch import tensor

"""
    >>> # Test 2 channels.
    >>> x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
    >>> y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0],[3.0, -2.0]]]])
    >>> fft_width = x.shape[-1]
    >>> fft_height = x.shape[-2]
    >>> pad_right = fft_width - y.shape[-1]
    >>> pad_bottom = fft_height - y.shape[-2]
    >>> y_padded = F.pad(y, (0, pad_right, 0, pad_bottom), 'constant', 0.0)
    >>> np.testing.assert_array_equal(x=tensor([
    ... [[[1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
    ... [[-1.0, 2.0, 0.0], [3.0, -2.0, 0.0], [0.0, 0.0, 0.0]]]]), y=y_padded,
    ... err_msg="The expected result x is different than the computed y.")
    >>> signal_ndim = 2
    >>> onesided = True
    >>> xfft = torch.rfft(x, signal_ndim=signal_ndim, onesided=onesided)
    >>> yfft = torch.rfft(y_padded, signal_ndim=signal_ndim, onesided=onesided)
    >>> result = correlate_fft_signals2D(xfft=xfft, yfft=yfft,
    ... input_height=fft_height, input_width=fft_width,
    ... half_fft_height=xfft.shape[-3], half_fft_width=xfft.shape[-2],
    ... out_height=(x.shape[-2]-y.shape[-2]+1),
    ... out_width=(x.shape[-1]-y.shape[-1] + 1))
    >>> # print("result: ", result)
    >>> np.testing.assert_array_almost_equal(
    ... x=np.array([[[[23.0, 32.0], [30., 4.]]]]),
    ... y=result, decimal=5,
    ... err_msg="The expected array x and computed y are not almost equal.")
"""

x = tensor([[[[-2.0000e+00, -1.0000e+00,  1.0000e+00, -2.0000e+00, -3.0000e+00],
[ 5.0000e+00,  2.0000e+00, -2.0000e+00,  1.0000e+00, -6.0000e+00],
[-1.0000e+00, -4.0000e+00,  1.0000e+00, -1.0000e+00, -3.0000e+00],
[ 1.7881e-07,  1.0000e+00, -7.0000e+00, -4.7684e-07, -3.0000e+00],
[ 5.0000e+00,  1.0000e+00,  2.0000e+00,  1.0000e+00, -1.0000e+00]],
[[-4.0000e+00,  1.0000e+00, -2.0000e+00, -2.0000e+00,  3.0000e+00],
[-3.0000e+00, -2.0000e+00, -1.0000e+00,  4.0000e+00, -2.0000e+00],
[ 5.0000e+00,  1.0000e+00, -3.0000e+00, -5.0000e+00,  2.0000e+00],
[-3.0000e+00, -5.0000e+00,  2.0000e+00, -1.0000e+00, -3.0000e+00],
[-6.0000e+00, -6.0000e+00, -1.0000e+00,  3.0000e+00, -8.0000e+00]]]])
