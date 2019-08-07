import torch
import numpy as np
from cnns.nnlib.robustness.channel_histograms.complex_mask import get_hyper_mask
from cnns.nnlib.utils.general_utils import next_power2
from torch.nn.functional import pad as torch_pad
from torch.distributions.laplace import Laplace

nprng = np.random.RandomState()
nprng.seed(31)


def fft_numpy(numpy_array, compress_rate):
    torch_image = torch.from_numpy(numpy_array).unsqueeze(dim=0)
    torch_image = fft_channel(input=torch_image, compress_rate=compress_rate)
    return torch_image.squeeze().cpu().numpy()


def fft_channel(input, compress_rate, val=0, get_mask=get_hyper_mask,
                onesided=True, is_next_power2=True):
    """
    In the forward pass we receive a Tensor containing the input
    and return a Tensor containing the output. ctx is a context
    object that can be used to stash information for backward
    computation. You can cache arbitrary objects for use in the
    backward pass using the ctx.save_for_backward method.

    :param input: the input image
    :param args: arguments that define: compress_rate - the compression
    ratio, interpolate - the interpolation within mask: const, linear, exp,
    log, etc.
    :param val: the value (to change coefficients to) for the mask
    :onesided: should use the onesided FFT thanks to the conjugate symmetry
    or want to preserve all the coefficients
    """
    # ctx.save_for_backward(input)
    # print("round forward")

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
    # assert H_fft == W_xfft, "The input tensor has to be squared."

    mask, _ = get_mask(H=H_xfft, W=W_xfft,
                       compress_rate=compress_rate,
                       val=val, interpolate='const',
                       onesided=onesided)
    mask = mask[:, 0:W_xfft, :]
    # print(mask)
    mask = mask.to(xfft.dtype).to(xfft.device)
    xfft = xfft * mask

    out = torch.irfft(input=xfft,
                      signal_ndim=2,
                      signal_sizes=(H_fft, W_fft),
                      onesided=onesided)
    out = out[..., :H, :W]
    return out


def gauss_noise_numpy(images, epsilon, bounds=(0, 1)):
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


def uniform_noise_numpy(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = nprng.uniform(low=-w, high=w, size=images.shape)
    return images + noise


def uniform_noise_torch(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = torch.zeros_like(images, requires_grad=False).uniform_(-w, w).to(
        images.device)
    return noise


def laplace_noise_numpy(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.laplace(loc=0, scale=scale, size=images.shape)
    return images + noise


def laplace_noise_torch(epsilon, images, bounds):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    distribution = Laplace(
        torch.zeros_like(images), torch.ones_like(images) * scale)
    return distribution.sample()


def beta_noise_numpy(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    w = epsilon * (max_ - min_)
    noise = nprng.beta(a=-w, b=w, size=images.shape)
    return images + noise

def logistic_noise_numpy(images, epsilon, bounds=(0, 1)):
    min_, max_ = bounds
    scale = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.logistic(loc=0, scale=scale, size=images.shape)
    return images + noise

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
    assert H == W
    index = int((1 - compress_rate / 100) * H)
    torch_compress_img = torch.zeros_like(torch_img)
    for c in range(C):
        try:
            u, s, v = torch.svd(torch_img[c])
        except RuntimeError as ex:
            print("SVD compression problem: ", ex)
            return None

        u_c = u[:, :index]
        s_c = s[:index]
        v_c = v[:, :index]

        torch_compress_img[c] = torch.mm(torch.mm(u_c, torch.diag(s_c)),
                                         v_c.t())
    return torch_compress_img


def compress_svd_numpy(numpy_array, compress_rate):
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
    C, H, W = torch_img.size()
    assert H == W
    index = int((1 - distort_rate / 100) * H)
    torch_compress_img = torch.zeros_like(torch_img)
    for c in range(C):
        try:
            u, s, v = torch.svd(torch_img[c])
        except RuntimeError as ex:
            print("SVD distortion problem: ", ex)
            return None

        u_c = u[:, index:]
        s_c = s[index:]
        v_c = v[:, index:]

        torch_compress_img[c] = torch.mm(torch.mm(u_c, torch.diag(s_c)),
                                         v_c.t())
    return torch_compress_img


def distort_svd_batch(x, distort_rate):
    result = torch.zeros_like(x)
    for i, torch_img in enumerate(x):
        result[i] = distort_svd(torch_img=torch_img,
                                distort_rate=distort_rate)
    return result
