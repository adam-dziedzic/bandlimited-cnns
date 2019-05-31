import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

figuresizex = 9.0
figuresizey = 6.1

# generate images
image1 = np.identity(5)
image2 = np.arange(16).reshape((4,4))



fig = plt.figure(figsize=(figuresizex,figuresizey))

# create your grid objects
top_row = ImageGrid(fig, 311, nrows_ncols = (1,3), axes_pad = .25,
                    cbar_location = "right", cbar_mode="single")
middle_row = ImageGrid(fig, 312, nrows_ncols = (1,3), axes_pad = .25,
                       cbar_location = "right", cbar_mode="single")
bottom_row = ImageGrid(fig, 313, nrows_ncols = (1,3), axes_pad = .25,
                       cbar_location = "right", cbar_mode="single")

# plot the images
for i in range(3):
    vmin, vmax = image1.min(),image1.max()
    ax = top_row[i]
    im1 = ax.imshow(image1, vmin=vmin, vmax=vmax)

for i in range(3):
    vmin, vmax = image2.min(),image2.max()
    ax =middle_row[i]
    im2 = ax.imshow(image2, vmin=vmin, vmax=vmax)

# Update showing how to use identical scale across all 3 images
# make some slightly different images and get their bounds
image2s = [image2,image2 + 5,image2 - 5]

# inelegant way to get the absolute upper and lower bounds from the three images
i_max, i_min = 0,0
for im in image2s:
    if im.max() > i_max:
        i_max= im.max()
    if im.min() < i_min:
        i_min = im.min()
# plot these as you would the others, but use identical vmin and vmax for all three fft_visualize
for i,im in enumerate(image2s):
    ax = bottom_row[i]
    im2_scaled = ax.imshow(im, vmin = i_min, vmax = i_max)

# add your colorbars
cbar1 = top_row.cbar_axes[0].colorbar(im1)
middle_row.cbar_axes[0].colorbar(im2)
bottom_row.cbar_axes[0].colorbar(im2_scaled)

# example of titling colorbar1
cbar1.set_label_text("label")

# readjust figure margins after adding colorbars,
# left and right are unequal because of how
# colorbar labels don't appear to factor in to the adjustment
plt.subplots_adjust(left=0.075, right=0.9)

plt.show()