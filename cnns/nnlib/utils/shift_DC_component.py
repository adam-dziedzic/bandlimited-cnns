import torch


def shift_to_center(xfft, onesided=True):
    W = xfft.size(-2)
    H = xfft.size(-3)

    HH = H // 2 + 1  # middle H
    WW = W // 2 + 1  # middle W only for bothsided (onesided = False)

    if onesided:
        top = xfft[..., :HH,:,:]
        bottom = xfft[..., HH:,:,:]
        xfft = torch.cat((top, bottom), dim=-3)
        return xfft
    else:
        top_left = xfft[..., :HH, :WW, :]
        top_right = xfft[..., :HH, WW:, :]
        bottom_left = xfft[..., HH:, :WW, :]
        bottom_right = xfft[..., HH:, WW:, :]
        top = torch.cat((top_right, top_left), dim=-2)
        bottom = torch.cat((bottom_right, bottom_left), dim=-2)
        return torch.cat((bottom, top), dim=-3)


def shift_to_center_np(xfft, onesided=True):
    xfft = torch.tensor(xfft)
    xfft = shift_to_center(xfft, onesided=onesided)
    return xfft.numpy()


def shift_to_corner(xfft, onesided=True):
    W = xfft.size(-2)
    H = xfft.size(-3)

    HH = H // 2  # This is on top here and should be moved back to the bottom.
    WW = W // 2

    if onesided:
        return torch.cat((xfft[..., HH:, ...],
                          xfft[..., :HH, ...]), dim=-2)
    else:
        return torch.cat(
            (xfft[..., HH:, :WW, ...], xfft[..., HH:, WW:, ...],
             xfft[..., :HH, :WW, ...], xfft[..., :HH, WW:, ...]), dim=-2)

def shift_to_corner_np(xfft, onesided=True):
    xfft = torch.tensor(xfft)
    xfft = shift_to_corner(xfft, onesided=onesided)
    return xfft.numpy()