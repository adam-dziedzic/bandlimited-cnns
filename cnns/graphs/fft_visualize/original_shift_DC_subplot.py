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
from cnns.nnlib.utils.complex_mask import get_disk_mask
from cnns.nnlib.utils.object import Object
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.utils.shift_DC_component import shift_DC
from mpl_toolkits.axes_grid1 import make_axes_locatable

# figuresizex = 10.0
# figuresizey = 10.0

# generate images

# dataset = "mnist"
dataset = "imagenet"

if dataset == "imagenet":
    limx, limy = 224, 224
elif dataset == "mnist":
    limx, limy = 28, 28

half = limx // 2
extent1 = [0, limx, 0, limy]
extent2 = [-half + 1, half, -half + 1, half]

images, labels = foolbox.utils.samples(dataset=dataset, index=0,
                                       batchsize=20,
                                       shape=(limx, limy),
                                       data_format='channels_first')
print("max value in images pixels: ", np.max(images))
images = images / 255
image = images[0]
label = labels[0]
print("label: ", label)
is_log = True


def process_fft(xfft):
    channel = 0
    xfft = xfft[channel, ...]
    xfft = xfft[:limx, :limy]
    return xfft


def get_fft(image, is_DC_shift=True):
    xfft = to_fft(image,
                  fft_type="magnitude",
                  is_log=is_log,
                  is_DC_shift=is_DC_shift,
                  onesided=False)
    return process_fft(xfft)


def show_fft(xfft, ax, extent=[0, limx, 0, limy]):
    vmin = xfft.min()
    # vmax = 110.0  # image.max()
    vmax = xfft.max()
    print("fft min, max: ", vmin, vmax)
    im = ax.imshow(xfft, vmin=vmin, vmax=vmax, cmap="hot",
                   interpolation='nearest', extent=extent)
    # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    return im


def show_image(image, ax, extent=[0, limx, 0, limy]):
    image = np.clip(image, a_min=0.0, a_max=1.0)
    image = np.moveaxis(image, 0, -1)
    vmin, vmax = image.min(), image.max()
    print("image min, max: ", vmin, vmax)
    image = np.squeeze(image)
    im = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest',
                   cmap="gray", extent=extent)
    divider = make_axes_locatable(ax)
    divider.append_axes('right', size='5%', pad=0.05, add_to_figure=False)
    return im


# type = "proxy"
# type = "original"
# type = "exact"
# type = "exact"
type = "dc"
# type = "practice"
# type = "sequence"
args = Arguments()
args.compress_fft_layer = 50
args.compress_rate = 50
args.next_power2 = False
args.is_DC_shift = True
result = Object()

if dataset == "mnist":
    image = np.expand_dims(image, 0)

xfft = get_fft(image)

image_exact = FFTBandFunctionComplexMask2D.forward(
    ctx=result,
    input=torch.from_numpy(image).unsqueeze(0),
    args=args,
    val=0,
    get_mask=get_hyper_mask,
    onesided=False).numpy().squeeze(0)
# These are complex numbers.
count_zeros = torch.sum(result.xfft == 0.0).item() / 2
count_all_vals = result.xfft.numel() / 2
print("% of zero-ed out exact: ", count_zeros / count_all_vals)
xfft_exact = result.xfft.squeeze(0)
xfft_exact = to_fft_magnitude(xfft_exact, is_log)
xfft_exact = process_fft(xfft_exact)

image_proxy = FFTBandFunction2D.forward(
    ctx=result,
    input=torch.from_numpy(image).unsqueeze(0),
    args=args,
    onesided=False).numpy().squeeze(0)
# These are complex numbers.
zero1 = torch.sum(result.xfft == 0.0).item() / 2
print("% of zero-ed out proxy: ", zero1 / (result.xfft.numel() / 2))
xfft_proxy = result.xfft.squeeze(0)
xfft_proxy = to_fft_magnitude(xfft_proxy, is_log)
xfft_proxy = process_fft(xfft_proxy)

# draw
figuresizex = 9.0
figuresizey = 8.1
fig = plt.figure(figsize=(figuresizex, figuresizey))
# fig = plt.figure()
cols = 2
if type == "sequence":
    cols = 4
# create your grid objects
rect = int(str(cols) + "11")
top_row = ImageGrid(fig, rect, nrows_ncols=(1, cols), axes_pad=.25,
                    cbar_location="right", cbar_mode="single")

if type == "proxy":
    image = image_proxy
    xfft = xfft_proxy
    ax = plt.subplot(1, cols, 1)
    show_image(image, ax, extent=extent1)
    ax = plt.subplot(1, cols, 2)
    im = show_fft(xfft, ax, extent=extent2)
elif type == "exact":
    image = image_exact
    xfft = xfft_exact
    ax = plt.subplot(1, cols, 1)
    show_image(image, ax, extent=extent1)
    ax = plt.subplot(1, cols, 2)
    im = show_fft(xfft, ax, extent=extent2)
elif type == "dc":
    xfft_center = get_fft(image)
    xfft_corner = get_fft(image, is_DC_shift=False)
    ax = plt.subplot(1, cols, 1)
    show_image(image, ax, extent=extent1)
    ax = plt.subplot(1, cols, 2)
    im = show_fft(xfft, ax, extent=extent2)
elif type == "practice":
    args.is_DC_shift = False
    result = Object()
    image_proxy = FFTBandFunction2D.forward(
        ctx=result,
        input=torch.from_numpy(image).unsqueeze(0),
        args=args,
        onesided=False).numpy().squeeze(0)
    show_image(image_proxy, index=0)
    xfft_proxy = result.xfft.squeeze(0)
    xfft_proxy = to_fft_magnitude(xfft_proxy, is_log)
    xfft_proxy = process_fft(xfft_proxy)
    im = show_fft(xfft_proxy, index=1)
elif type == "sequence":
    show_image(image, index=0)
    show_fft(xfft_exact, index=1)
    show_fft(xfft_proxy, index=2)
    args.is_DC_shift = False
    result = Object()
    image_proxy = FFTBandFunction2D.forward(
        ctx=result,
        input=torch.from_numpy(image).unsqueeze(0),
        args=args,
        onesided=False).numpy().squeeze(0)

    xfft_proxy = result.xfft.squeeze(0)
    xfft_proxy = to_fft_magnitude(xfft_proxy, is_log)
    xfft_proxy = process_fft(xfft_proxy)
    im = show_fft(xfft_proxy, index=3)
else:
    raise Exception(f"Unknown type: {type}")

# add your colorbars
cbar1 = top_row.cbar_axes[0].colorbar(im)

# example of titling colorbar1
cbar1.set_label_text("20*log10")

# readjust figure margins after adding colorbars,
# left and right are unequal because of how
# colorbar labels don't appear to factor in to the adjustment
plt.subplots_adjust(left=0.075, right=0.9)

format = "png"  # "png" "pdf"
plt.tight_layout()
plt.show(block=True)
plt.savefig(fname="images/" + type + "-" + dataset + "3." + format,
            format=format, transparent=True)
plt.close()
