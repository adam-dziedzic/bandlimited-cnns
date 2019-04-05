import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

lim_x = 224
lim_y = 224

lim_x = 10
lim_y = 10


dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "original_fft.csv.npy")
original_fft = np.load(output_path)

cmap_type = "custom"
vmin_heatmap = -6
vmax_heatmap = 10
labels = "Text" # "None" or "Text

if cmap_type == "custom":
    # setting for the heat map
    # cdict = {
    #     'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
    #     'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
    #     'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
    # }

    cdict = {'red': [(0.0, 0.0, 0.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 1.0, 1.0)],

             'green': [(0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 1.0, 1.0)],

             'blue': [(0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 1.0, 1.0)]}

    # cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
    # cmap = "hot"
    # cmap = "YlGnBu"
    # cmap = 'PuBu_r'
    # cmap = "seismic"
    # cmap_type = 'OrRd'

    x = np.arange(0, lim_x, 1.)
    y = np.arange(0, lim_y, 1.)
    X, Y = np.meshgrid(x, y)
elif cmap_type == "standard":
    # https://matplotlib.org/tutorials/colors/colormaps.html
    # cmap = 'hot'
    # cmap = 'rainbow'
    # cmap = 'seismic'
    # cmap = 'terrain'
    cmap = 'OrRd'
    interpolation = 'nearest'
else:
    raise Exception(f"Unknown type of the cmap: {cmap_type}.")

np.save(output_path, original_fft)

# go back to the original print size
# np.set_printoptions(threshold=options['threshold'])
original_fft = original_fft[:lim_y, :lim_x]

if cmap_type == "standard":
    plt.imshow(original_fft, cmap=cmap,
               interpolation=interpolation)
    heatmap_legend = plt.pcolor(original_fft)
    plt.colorbar(heatmap_legend)
elif cmap_type == "custom":
    fig, ax = plt.subplots()
    # plt.pcolor(X, Y, original_fft, cmap=cmap, vmin=vmin_heatmap,
    #            vmax=vmax_heatmap)
    cax = ax.matshow(original_fft, cmap='seismic', vmin=vmin_heatmap,
               vmax=vmax_heatmap)
    # plt.colorbar()
    fig.colorbar(cax)
    if labels == "Text":
        for (i, j), z in np.ndenumerate(original_fft):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

channel = 0
# plt.axis('off')
plt.ylabel("fft-ed\nchannel " + str(channel))
plt.show(block=True)
