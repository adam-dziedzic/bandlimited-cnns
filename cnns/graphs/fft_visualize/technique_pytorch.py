import numpy as np
import torch
from cnns.nnlib.utils.shift_DC_component import shift_DC
from cnns.nnlib.pytorch_layers.pytorch_utils import compress_2D_index_forward

torch.manual_seed(31)
# x = torch.randint(10, (6, 6))
x = torch.tensor([[6., 5., 0., 2., 4., 1.],
        [4., 2., 8., 5., 6., 8.],
        [0., 0., 4., 2., 0., 2.],
        [2., 9., 9., 9., 1., 6.],
        [3., 0., 9., 4., 6., 6.],
        [7., 3., 4., 7., 9., 0.]])
x = x.to(torch.float)
print("sum of x: ", x.sum().item())
print("x: ", x)

xfft = torch.rfft(x, onesided=False, signal_ndim=2, normalized=False)
print("xfft: ", xfft)

# xfft:  tensor([[[153.0000,   0.0000],
#          [-16.0000,  -3.4641],
#          [  0.0000,  10.3923],
#          [ 11.0000,   0.0000],
#          [  0.0000, -10.3923],
#          [-16.0000,   3.4641]],
#
#         [[ -4.5000,  14.7224],
#          [  7.5000,   7.7942],
#          [ 18.0000, -12.1244],
#          [ 16.5000,  12.9904],
#          [ -1.5000,  23.3827],
#          [ 12.0000, -15.5885]],
#
#         [[  4.5000, -19.9186],
#          [ 14.0000, -12.1244],
#          [ 16.5000,   0.8660],
#          [-20.5000,  -0.8660],
#          [-12.0000,  19.0526],
#          [  3.5000,  12.9904]],
#
#         [[-45.0000,   0.0000],
#          [  9.0000,   5.1962],
#          [ -3.0000,   1.7321],
#          [  9.0000,   0.0000],
#          [ -3.0000,  -1.7321],
#          [  9.0000,  -5.1962]],
#
#         [[  4.5000,  19.9186],
#          [  3.5000, -12.9904],
#          [-12.0000, -19.0526],
#          [-20.5000,   0.8660],
#          [ 16.5000,  -0.8660],
#          [ 14.0000,  12.1244]],
#
#         [[ -4.5000, -14.7224],
#          [ 12.0000,  15.5885],
#          [ -1.5000, -23.3827],
#          [ 16.5000, -12.9904],
#          [ 18.0000,  12.1244],
#          [  7.5000,  -7.7942]]])

xfft_dc = shift_DC(xfft, onesided=False)
print("xfft_dc: ", xfft_dc)

xfft = torch.rfft(x, onesided=True, signal_ndim=2, normalized=False)
print("xfft onesided: ", xfft)

xfft_dc = shift_DC(xfft, onesided=True)
print("xfft dc onesided: ", xfft_dc)

xfft_compress1 = compress_2D_index_forward(xfft, index_forward=3)
xfft_compress1_dc = shift_DC(xfft_compress1, onesided=True)
print("xfft compress1: ", xfft_compress1_dc)

xfft_compress2 = compress_2D_index_forward(xfft, index_forward=2)
xfft_compress2_dc = shift_DC(xfft_compress2, onesided=True)
print("xfft compress2: ", xfft_compress2_dc)
