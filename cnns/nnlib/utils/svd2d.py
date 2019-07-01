import torch

def compress_svd(torch_img, compress_rate):
    C, H, W = torch_img.size()
    assert H == W
    index = int((1 - compress_rate) * H)
    torch_compress_img = torch.zeros_like(torch_img)
    for c in range(C):
        u, s, v = torch.svd(torch_img[c])

        u_c = u[:, :index]
        s_c = s[:index]
        v_c = v[:, :index]

        torch_compress_img[c] = torch.mm(torch.mm(u_c, torch.diag(s_c)),
                                         v_c.t())
    return torch_compress_img
