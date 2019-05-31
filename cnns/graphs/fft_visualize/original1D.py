import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import foolbox
from cnns.nnlib.robustness.utils import to_fft
from cnns.nnlib.robustness.utils import to_fft_magnitude
import torch
from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBandFunction2D
from cnns.nnlib.pytorch_layers.fft_band_2D_complex_mask import \
    FFTBandFunctionComplexMask2D
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.utils.object import Object
from cnns.nnlib.datasets.ucr.ucr_example import fifty_words

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_BLACK, MY_ORANGE, MY_GOLD, MY_RED]]

figuresizex = 9.0
figuresizey = 5.0
fig = plt.figure(figsize=(figuresizex, figuresizey))

lw = 1
onesided = False

signal = fifty_words

print("min max value signal: ", np.min(signal), np.max(signal))
is_log = True
markers = ["^", "o", "v", "s", "D", "p"]

signal = signal[0][0]
print("signal length: ", len(signal))
plt.subplot(2, 2, 1)
plt.plot(range(len(signal)), signal, label="input signal", lw=1,
         color=colors[1])
plt.title("Time domain")
plt.ylabel("Amplitude")
plt.xlabel("Sample number")
plt.legend(frameon=False)

plt.subplot(2, 2, 2)

xfft = torch.rfft(torch.from_numpy(signal), onesided=onesided, signal_ndim=1)
xfft_mag = to_fft_magnitude(xfft, is_log)
plt.plot(range(len(signal)), xfft_mag, label="fft-ed signal", lw=lw,
         color=colors[2])
plt.title("Frequency domain")
plt.ylabel("dB")
plt.xlabel("Frequency (sample number)")
plt.legend(frameon=False)

compression_rate = 95
out_len = len(signal) * (100 - compression_rate) / 100
start = int(out_len // 2)
end = int(len(signal) - start)
print("compression rate, start, end: ", compression_rate, start, end)
xfft[start:end] = 0
signal2 = torch.irfft(xfft, onesided=onesided, signal_ndim=1)
signal2 = signal2.numpy()

plt.subplot(2, 2, 3)
plt.plot(range(len(signal2)), signal, label="input signal", lw=lw,
         color=colors[1])
plt.plot(range(len(signal2)), signal2, label="compressed signal\n(" + str(
    compression_rate) + "% compression\nrate)", lw=lw,
         color=colors[5])
plt.legend(loc="upper left", frameon=False)
plt.ylabel("Amplitude")
plt.xlabel("Sample number")

plt.subplot(2, 2, 4)
xfft_mag = to_fft_magnitude(xfft, is_log)
plt.plot(range(len(signal)), xfft_mag,
         label="fft-ed signal\n(" + str(
             compression_rate) + "% compression rate)",
         lw=lw, color=colors[2])
plt.ylabel("dB")
plt.xlabel("Frequency (sample number)")
plt.legend(loc="upper center", frameon=False)

# type = "exact"
# type = "proxy"
compress_fft_layer = 50
result = Object()

# xfft = get_fft(image)
# image = np.clip(image, a_min=0.0, a_max=1.0)

# xfft = get_fft(image)
# xfft += np.abs(xfft.min())
# xfft = np.clip(xfft, a_min=0.0, a_max=118.0)
# image = np.moveaxis(image, 0, -1)


# example of titling colorbar1
# cbar1.set_label_text("label")

# readjust figure margins after adding colorbars,
# left and right are unequal because of how
# colorbar labels don't appear to factor in to the adjustment
# plt.subplots_adjust(left=0.075, right=0.9)

plt.subplots_adjust(hspace=0.3)

type = "original"
format = "pdf"  # "png" "pdf"
plt.savefig(fname=type + "-1D." + format, format=format)
plt.show(block=True)
plt.close()
