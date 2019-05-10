import torch


def shift_DC(xfft, onesided=True, shift_type="center"):
    W = xfft.size(-2)
    H = xfft.size(-3)

    if shift_type == "center":
        HH = H // 2 + 1  # middle H
        WW = W // 2 + 1  # middle W only for bothsided (onesided = False)
    else:
        # This is on top here and should be moved back to the bottom.
        HH = H // 2
        WW = W // 2

    if onesided:
        top = xfft[..., :HH, :, :]
        bottom = xfft[..., HH:, :, :]
        return torch.cat((bottom, top), dim=-3)
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
    xfft = shift_DC(xfft, onesided=onesided)
    return xfft.numpy()


def shift_to_corner(xfft, onesided=True):
    W = xfft.size(-2)
    H = xfft.size(-3)

    # This is on top here and should be moved back to the bottom.
    HH = H // 2
    WW = W // 2

    if onesided:
        top = xfft[..., :HH, :, :]
        bottom = xfft[..., HH:, :, :]
        return torch.cat((bottom, top), dim=-3)
    else:
        top_left = xfft[..., :HH, :WW, :]
        top_right = xfft[..., :HH, WW:, :]
        bottom_left = xfft[..., HH:, :WW, :]
        bottom_right = xfft[..., HH:, WW:, :]
        top = torch.cat((top_right, top_left), dim=-2)
        bottom = torch.cat((bottom_right, bottom_left), dim=-2)
        return torch.cat((bottom, top), dim=-3)


def shift_to_corner_np(xfft, onesided=True):
    xfft = torch.tensor(xfft)
    xfft = shift_to_corner(xfft, onesided=onesided)
    return xfft.numpy()
