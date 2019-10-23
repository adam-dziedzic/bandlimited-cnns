import torch
import numpy as np
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.utils.complex_mask import get_inverse_hyper_mask
from cnns.nnlib.utils.general_utils import next_power2
from torch.nn.functional import pad as torch_pad
from torch.distributions.laplace import Laplace
from cnns.nnlib.pytorch_layers.pytorch_utils import get_xfft_hw
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
import functools
from numpy.linalg import svd
import math

nprng = np.random.RandomState()
nprng.seed(31)


def numpy_decorator(call_fn):
    @functools.wraps(call_fn)
    def wrapper(numpy_array, **kwargs):
        # Unsqueeze for batch processing.
        torch_tensor = torch.from_numpy(numpy_array).unsqueeze(dim=0)
        torch_result = call_fn(torch_tensor, **kwargs)
        return torch_result.squeeze().detach().cpu().numpy()

    return wrapper


def fft_numpy(numpy_array, compress_rate, inverse_compress_rate=0):
    torch_image = torch.from_numpy(numpy_array).unsqueeze(dim=0)
    torch_image = fft_channel(input=torch_image, compress_rate=compress_rate,
                              inverse_compress_rate=inverse_compress_rate)
    return torch_image.squeeze().cpu().numpy()


def fft_zero_values_numpy(numpy_array, high, low=0, onesided=True,
                          is_next_power2=True):
    torch_image = torch.from_numpy(numpy_array).unsqueeze(dim=0)
    torch_image = fft_zero_values(
        input=torch_image, high=high, low=low, onesided=onesided,
        is_next_power2=is_next_power2)
    return torch_image.squeeze().cpu().numpy()


def fft_channel(input, compress_rate, val=0, get_mask=get_hyper_mask,
                onesided=True, is_next_power2=False, inverse_compress_rate=0,
                get_inv_mask=get_inverse_hyper_mask):
    """
    :param input: the input image
    :compress_rate: percentage of high frequency coefficients that should be
    removed
    :get_mask: the mask to discard the frequency coefficients (this can be a
    squared mask that is inexact or a "hyperbolic" one which is exact).
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    :is_next_power2: should we bring the FFT size to the power of 2 (usually for
    faster computation but with higher memory usage).
    :inverse_compress_rate: compress by removing low frequency coefficients
    :get_inv_mask: the inverted mask for the removal of low frequency
    coefficients
    """
    N, C, H, W = input.size()

    if H != W:
        raise Exception("We support only squared input.")

    if is_next_power2:
        H_fft = next_power2(H)
        W_fft = next_power2(W)
        pad_H = H_fft - H
        pad_W = W_fft - W
        input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)
    else:
        H_fft = H
        W_fft = W
    xfft = torch.rfft(input,
                      signal_ndim=2,
                      onesided=onesided)
    del input

    _, _, H_xfft, W_xfft, _ = xfft.size()

    mask, _ = get_mask(H=H_xfft, W=W_xfft,
                       compress_rate=compress_rate,
                       val=val, interpolate='const',
                       onesided=onesided)

    if inverse_compress_rate > 0 and get_inv_mask is not None:
        inv_mask, _ = get_inv_mask(H=H_xfft, W=W_xfft,
                                   compress_rate=inverse_compress_rate,
                                   val=val, interpolate='const',
                                   onesided=onesided)
        mask = mask + inv_mask

    mask = mask[:, 0:W_xfft, :]
    mask = mask.to(xfft.dtype).to(xfft.device)
    xfft = xfft * mask

    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def fft_squared_channel(input, compress_rate, onesided=True,
                        is_next_power2=False):
    """
    Simple implementation of the fft channel that removes the coefficients in
    L-shaped fashion starting from the highest frequency coefficients. This
    method is aimed at high compressibility.

    :param input: the input image
    :compress_rate: percentage of high frequency coefficients that should be
    removed
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    :is_next_power2: should we bring the FFT size to the power of 2 (usually for
    faster computation but with higher memory usage).
    """
    N, C, H, W = input.size()

    if H != W:
        raise Exception("We support only squared input.")

    if is_next_power2:
        H_fft = next_power2(H)
        W_fft = next_power2(W)
        pad_H = H_fft - H
        pad_W = W_fft - W
        input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)
    else:
        H_fft = H
        W_fft = W

    xfft = torch.rfft(input,
                      signal_ndim=2,
                      onesided=onesided)
    del input

    _, _, H_xfft, W_xfft, _ = xfft.size()

    # Remove the coefficients in the L-shaped fashion.
    if onesided:
        retain_rate = 1 - compress_rate / 100
        n = int(math.sqrt(retain_rate * H_xfft * W_xfft / 2) - 1)
    else:
        raise Exception(
            'Compression not supported for full FFT representation.')
    xfft[..., n + 1:-n, :, :] = 0  # zero out center
    xfft[..., :, n + 1:, :] = 0  # zero out right-end stripe

    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def fft_zero_values(input, high, low=0, onesided=True, is_next_power2=True):
    """
    :param input: the input image
    :param high: the highest value to be zeroed out (up to)
    :param low: the lowest value to be zeroed out (down to)
    :param val: the value (to change coefficients to) for the mask
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    :is_next_power2: should we bring the FFT size to the power of 2
    :inverse_compress_rate:
    :return the zero out specific values
    """
    _, _, H, W = input.size()
    xfft, H_fft, W_fft = get_xfft_hw(input=input, signal_ndim=2,
                                     onesided=onesided,
                                     is_next_power2=is_next_power2)
    del input

    _, _, H_xfft, W_xfft, _ = xfft.size()
    # assert H_fft == W_xfft, "The input tensor has to be squared."
    mask = torch.zeros_like(xfft)
    xfft_abs = torch.abs(xfft)
    mask[xfft_abs < low] = 1.0
    mask[xfft_abs > high] = 1.0
    xfft *= mask

    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def fft_zero_low_magnitudes(input, high, low=0, onesided=True,
                            is_next_power2=True):
    """
    :param input: the input image
    :param high: the highest value to be zeroed out (up to)
    :param low: the lowest value to be zeroed out (down to)
    :param val: the value (to change coefficients to) for the mask
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    :is_next_power2: should we bring the FFT size to the power of 2
    :inverse_compress_rate:
    :return the zero out specific values corresponding to low magnitudes
    """
    _, _, H, W = input.size()
    xfft, H_fft, W_fft = get_xfft_hw(input=input, signal_ndim=2,
                                     onesided=onesided,
                                     is_next_power2=is_next_power2)
    del input

    _, _, H_xfft, W_xfft, _ = xfft.size()
    # assert H_fft == W_xfft, "The input tensor has to be squared."
    spectrum = get_spectrum(xfft, squeeze=False)
    mask = torch.zeros_like(spectrum)
    mask[spectrum < low] = 1.0
    mask[spectrum > high] = 1.0
    # Extend the mask to cover two parts of the complex numbers in the last dim.
    mask = torch.cat(tensors=(mask, mask), dim=-1)
    xfft *= mask
    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def replace_frequencies_numpy(input_to, input_from, compress_rate, val=0,
                              get_mask=get_hyper_mask, high=True,
                              onesided=True, is_next_power2=True):
    # Copy the input_to without sharing memory.
    torch_image_to = torch.tensor(input_to).unsqueeze(dim=0)
    # We only copy from input_from so the memory might be shared with tensor.
    torch_image_from = torch.from_numpy(input_from).unsqueeze(dim=0)
    torch_image = replace_frequencies(
        input_to=torch_image_to,
        input_from=torch_image_from,
        compress_rate=compress_rate,
        val=val,
        get_mask=get_mask,
        high=high,
        onesided=onesided,
        is_next_power2=is_next_power2)
    return torch_image.squeeze().cpu().numpy()


def replace_frequencies(input_to, input_from, compress_rate, val=0,
                        get_mask=get_hyper_mask, high=True,
                        onesided=True, is_next_power2=True):
    """
    Replace the high frequencies in input_to with the high_frequencies from
    input_from.

    :param input: the input image
    :param args: arguments that define: compress_rate - the compression
    ratio, interpolate - the interpolation within mask: const, linear, exp,
    log, etc.
    :param val: the value (to change coefficients to) for the mask
    :param get_mask: function to generate a mask
    :param high: replace high or low frequencies
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    :is_next_power2: should we bring the FFT size to the power of 2
    :inverse_compress_rate:
    """
    # ctx.save_for_backward(input)
    # print("round forward")

    N, C, H, W = input_to.size()
    (NN, CC, HH, WW) = input_from.size()
    assert N == NN and C == CC and H == HH and W == WW, 'The inputs are not of the same size.'

    if H != W:
        raise Exception("Support provided only squared input.")

    if is_next_power2:
        H_fft = next_power2(H)
        W_fft = next_power2(W)
        pad_H = H_fft - H
        pad_W = W_fft - W
        input_to = torch_pad(input_to, [0, pad_W, 0, pad_H], 'constant', 0)
        input_from = torch_pad(input_from, [0, pad_W, 0, pad_H], 'constant', 0)
    else:
        H_fft = H
        W_fft = W

    xfft_to = torch.rfft(input_to,
                         signal_ndim=2,
                         onesided=onesided)
    xfft_from = torch.rfft(input_from,
                           signal_ndim=2,
                           onesided=onesided)
    del input_to
    del input_from

    _, _, H_xfft, W_xfft, _ = xfft_to.size()

    mask, _ = get_mask(H=H_xfft, W=W_xfft,
                       compress_rate=compress_rate,
                       val=val, interpolate='const',
                       onesided=onesided)

    inv_mask = mask * (-1) + 1

    mask = mask[:, 0:W_xfft, :]
    mask = mask.to(xfft_to.dtype).to(xfft_to.device)

    inv_mask = inv_mask[:, 0:W_xfft, :]
    inv_mask = inv_mask.to(xfft_from.dtype).to(xfft_from.device)

    if high:
        xfft_to = xfft_to * mask
        xfft_from = xfft_from * inv_mask
    else:
        xfft_to = xfft_to * inv_mask
        xfft_from = xfft_from * mask

    xfft_to = xfft_to + xfft_from

    out = torch.irfft(input=xfft_to,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def gauss_noise_numpy(images, epsilon, bounds=(0, 1)):
    # if epsilon == 0:
    #     return images.copy()
    min_, max_ = bounds
    std = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.normal(scale=std, size=images.shape)
    return images + noise


def gauss_noise_torch(epsilon, images, bounds=(0, 1)):
    min_, max_ = bounds
    std = epsilon / np.sqrt(3) * (max_ - min_)
    noise = torch.zeros_like(images, requires_grad=False).normal_(0, std).to(
        images.device)
    return noise


def gauss_noise_fft_torch(std, input, is_next_power2=False, onesided=True):
    N, C, H, W = input.size()

    if H != W:
        raise Exception("We support only squared input.")

    if is_next_power2:
        H_fft = next_power2(H)
        W_fft = next_power2(W)
        pad_H = H_fft - H
        pad_W = W_fft - W
        input = torch_pad(input, (0, pad_W, 0, pad_H), 'constant', 0)
    else:
        H_fft = H
        W_fft = W
    xfft = torch.rfft(input,
                      signal_ndim=2,
                      onesided=onesided)
    del input

    noise = torch.zeros_like(xfft, requires_grad=False).normal_(0, std).to(
        xfft.device)

    # xfft += xfft * noise
    xfft += noise

    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def uniform_noise_numpy(images, epsilon, bounds=(0, 1), op=np.add):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = nprng.uniform(low=-w, high=w, size=images.shape)
    return op(images, noise)


def uniform_noise_torch(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = torch.zeros_like(images, requires_grad=False).uniform_(-w, w).to(
        images.device)
    return noise


def laplace_noise_numpy(images, epsilon, bounds=(0, 1), op=np.add):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.laplace(loc=0, scale=scale, size=images.shape)
    return op(images, noise)


def laplace_noise_numpy_subtract(images, epsilon, bounds=(0, 1)):
    return laplace_noise_numpy(images=images,
                               epsilon=epsilon,
                               bounds=bounds,
                               op=np.subtract)


def laplace_noise_torch(epsilon, images, bounds):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    distribution = Laplace(
        torch.zeros_like(images), torch.ones_like(images) * scale)
    return distribution.sample()


def beta_noise_numpy(images, epsilon, bounds=(0, 1), op=np.add):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = nprng.beta(a=-w, b=w, size=images.shape)
    return op(images, noise)


def logistic_noise_numpy(images, epsilon, bounds=(0, 1), op=np.add):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.logistic(loc=0, scale=scale, size=images.shape)
    return op(images, noise)


def round(values_per_channel, images):
    round_multiplier = values_per_channel - 1.0
    ext_multiplier = 1.0 / round_multiplier
    return ext_multiplier * torch.round(round_multiplier * images)


def round_numpy(image, values_per_channel=16):
    round_multiplier = values_per_channel - 1.0
    ext_multiplier = 1.0 / round_multiplier
    return ext_multiplier * np.around(round_multiplier * image)


def subtract_rgb(images, subtract_value):
    values_per_channel = 256
    round_multiplier = values_per_channel - 1.0
    # from [0,1] to [0,255]
    images = torch.round(round_multiplier * images)
    images = images - subtract_value
    ext_multiplier = 1.0 / round_multiplier
    return ext_multiplier * images


def subtract_rgb_numpy(images, subtract_value):
    values_per_channel = 256
    round_multiplier = values_per_channel - 1.0
    # from [0,1] to [0,255]
    images = np.around(round_multiplier * images)
    images = images - subtract_value
    ext_multiplier = 1.0 / round_multiplier
    return ext_multiplier * images


def compress_svd(torch_img, compress_rate):
    C, H, W = torch_img.size()
    D = min(H, W)
    index = int((1 - compress_rate / 100) * D)
    torch_compress_img = torch.zeros_like(torch_img)
    for c in range(C):
        iters = 10
        for i in range(iters):
            try:
                u, s, v = torch.svd(torch_img[c])
                u_c = u[:, :index]
                s_c = s[:index]
                v_c = v[:, :index]
                torch_compress_img[c] = torch.mm(torch.mm(u_c, torch.diag(s_c)),
                                                 v_c.t())
                break
            except RuntimeError as ex:
                msg = "SVD compression problem: ", ex, " iteration: ", i
                print(msg)

                if i == (iters - 1):
                    torch_compress_img[c] = torch_img[c]

    return torch_compress_img


def compress_svd_numpy_through_torch(numpy_array, compress_rate):
    torch_image = torch.from_numpy(numpy_array)
    torch_image = compress_svd(torch_img=torch_image,
                               compress_rate=compress_rate)
    return torch_image.cpu().numpy()


def compress_svd_batch(x, compress_rate):
    result = torch.zeros_like(x)
    for i, torch_img in enumerate(x):
        result[i] = compress_svd(torch_img=torch_img,
                                 compress_rate=compress_rate)
    return result


def distort_svd(torch_img, distort_rate):
    return compress_svd(torch_img=torch_img, compress_rate=distort_rate)


def distort_svd_batch(x, distort_rate):
    return compress_svd_batch(x=x, compress_rate=distort_rate)


def compress_svd_numpy(numpy_array, compress_rate):
    H = numpy_array.shape[-2]
    W = numpy_array.shape[-1]
    c = compress_rate / 100
    """
    (1-c) = (2*H*index + index) / (H * W)
    (1-c) = index * (2*H + 1) / (H * W)
    index = (1-c) * (H * W) / (2*H + 1) 
    """
    index = int((1 - c) * (H * W) / (2 * H + 1))
    iters = 10
    compress_img = numpy_array
    for i in range(iters):
        try:
            u, s, vh = svd(a=numpy_array, full_matrices=False)
            u_c = u[..., :index]
            s_c = s[..., :index]
            vh_c = vh[..., :index, :]
            compress_img = np.matmul(u_c * s_c[..., None, :], vh_c)
            break
        except RuntimeError as ex:
            msg = "SVD compression problem: ", ex, " iteration: ", i
            print(msg)

    return compress_img


def compress_svd_through_numpy(tensor, compress_rate):
    device = tensor.device
    numpy_image = tensor.cpu().numpy()
    numpy_image = compress_svd_numpy(numpy_array=numpy_image,
                                     compress_rate=compress_rate)
    return torch.from_numpy(numpy_image).to(device)


def compress_svd_resize(numpy_array, compress_rate):
    H = numpy_array.shape[-2]
    W = numpy_array.shape[-1]
    c = compress_rate / 100
    """
    (1-c) = (index*index) / (H * W)
    index = sqrt((1-c) * (H * W)) 
    """
    index = int(math.sqrt((1 - c) * (H * W)))
    iters = 10
    compress_img = numpy_array
    for i in range(iters):
        try:
            u, s, vh = svd(a=numpy_array, full_matrices=False)
            u = u[..., :index]
            s[..., (index - 1):] = 0
            vh = vh[..., :index, :]
            # compress_img = np.matmul(u_c * s_c[..., None, :], vh_c)
            compress_img = np.matmul(vh * s[..., None, :], u)
            break
        except RuntimeError as ex:
            msg = "SVD compression problem: ", ex, " iteration: ", i
            print(msg)

    return compress_img


def compress_svd_resize_through_numpy(tensor, compress_rate):
    device = tensor.device
    numpy_image = tensor.cpu().numpy()
    numpy_image = compress_svd_resize(numpy_array=numpy_image,
                                      compress_rate=compress_rate)
    return torch.from_numpy(numpy_image).to(device)


def to_svd_numpy(numpy_array, compress_rate):
    """
    We transform an image to its SVD representation: U x D x V^T.
    We return as output for each channel X, 2 channels with values: U x D, and V^T.
    The initial size is n^2, the final size is: 2np

    :param numpy_array: the input image
    :param compress_rate: the compression rate
    :return: the image with 2 times more channels
    """
    H = numpy_array.shape[-2]
    W = numpy_array.shape[-1]
    assert H == W
    c = compress_rate / 100
    """
    (1-c) = (2*H*index) / H * W
    (1-c) = 2*index / W
    index = (1-c) * W / 2
    """
    index = int((1 - c) * W / 2)
    try:
        u, s, vh = svd(a=numpy_array, full_matrices=False)
        u_c = u[..., :index]
        s_c = s[..., :index]
        vh_c = vh[..., :index, :]

        u_s = u_c * s_c[..., None, :]
        v_s = vh_c.transpose((0, 2, 1))
        result = np.concatenate((u_s, v_s), axis=0)
    except RuntimeError as ex:
        msg = "SVD compression problem: " + ex
        print(msg)
        raise ex

    return result


def to_svd_through_numpy(tensor, compress_rate):
    device = tensor.device
    numpy_image = tensor.cpu().numpy()
    numpy_image = to_svd_numpy(numpy_array=numpy_image,
                               compress_rate=compress_rate)
    return torch.from_numpy(numpy_image).to(device)
