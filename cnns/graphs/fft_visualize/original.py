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

figuresizex = 9.0
figuresizey = 8.1

# figuresizex = 10.0
# figuresizey = 10.0

# generate images
# image1 = np.identity(5)
image2 = np.arange(16).reshape((4, 4))

# dataset = "mnist"
dataset = "imagenet"

if dataset == "imagenet":
    limx, limy = 224, 224
elif dataset == "mnist":
    limx, limy = 28, 28

images, labels = foolbox.utils.samples(dataset=dataset, index=0,
                                       batchsize=20,
                                       shape=(limx, limy),
                                       data_format='channels_first')
print("max value in images pixels: ", np.max(images))
images = images / 255
image = images[0]
is_log = True


def process_fft(xfft):
    channel = 0
    xfft = xfft[channel, ...]
    xfft = xfft[:limx, :limy]
    return xfft


def get_fft(image):
    xfft = to_fft(image, fft_type="magnitude", is_log=is_log)
    return process_fft(xfft)


type = "original"
# type = "exact"
# type = "proxy"
compress_fft_layer = 50
result = Object()

if dataset == "mnist":
    image = np.expand_dims(image, 0)

xfft = get_fft(image)

image_exact = FFTBandFunctionComplexMask2D.forward(
    ctx=result,
    input=torch.from_numpy(image).unsqueeze(0),
    compress_rate=compress_fft_layer,
    val=0,
    interpolate="const",
    get_mask=get_hyper_mask,
    onesided=False).numpy().squeeze(0)
xfft_exact = result.xfft.squeeze(0)
xfft_exact = to_fft_magnitude(xfft_exact, is_log)
xfft_exact = process_fft(xfft_exact)

image_proxy = FFTBandFunction2D.forward(
    ctx=result,
    input=torch.from_numpy(image).unsqueeze(0),
    compress_rate=compress_fft_layer,
    onesided=False).numpy().squeeze(0)
xfft_proxy = result.xfft.squeeze(0)
xfft_proxy = to_fft_magnitude(xfft_proxy, is_log)
xfft_proxy = process_fft(xfft_proxy)

image = np.clip(image, a_min=0.0, a_max=1.0)

# xfft = get_fft(image)
# xfft += np.abs(xfft.min())
# xfft = np.clip(xfft, a_min=0.0, a_max=118.0)

image = np.moveaxis(image, 0, -1)

fig = plt.figure(figsize=(figuresizex, figuresizey))

cols = 4
# create your grid objects
top_row = ImageGrid(fig, 411, nrows_ncols=(1, cols), axes_pad=.25,
                    cbar_location="right", cbar_mode="single")

# plot the image
vmin, vmax = image.min(), image.max()
print("image min, max: ", vmin, vmax)
ax = top_row[0]
image = np.squeeze(image)
im1 = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest',
                cmap="gray")

# plot the FFT map
image = xfft
vmin = image.min()
# vmax = 110.0  # image.max()
vmax = image.max()
print("fft min, max: ", vmin, vmax)
ax = top_row[1]
im1 = ax.imshow(image, vmin=vmin, vmax=vmax, cmap="hot",
                interpolation='nearest')

# plot the FFT exact
image = xfft_exact
vmin = image.min()
# vmax = 110.0  # image.max()
vmax = image.max()
print("fft min, max: ", vmin, vmax)
ax = top_row[2]
im1 = ax.imshow(image, vmin=vmin, vmax=vmax, cmap="hot",
                interpolation='nearest')

# plot the FFT proxy
image = xfft_proxy
vmin = image.min()
# vmax = 110.0  # image.max()
vmax = image.max()
print("fft min, max: ", vmin, vmax)
ax = top_row[3]
im1 = ax.imshow(image, vmin=vmin, vmax=vmax, cmap="hot",
                interpolation='nearest')

# add your colorbars
cbar1 = top_row.cbar_axes[0].colorbar(im1)

# example of titling colorbar1
# cbar1.set_label_text("label")

# readjust figure margins after adding colorbars,
# left and right are unequal because of how
# colorbar labels don't appear to factor in to the adjustment
plt.subplots_adjust(left=0.075, right=0.9)

format = "png"  # "png" "pdf"
plt.savefig(fname="images/" + type + "-" + dataset + "-4cols." + format,
            format=format, transparent=True)
plt.show(block=True)
plt.close()
