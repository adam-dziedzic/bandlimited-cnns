import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import time

start = time.time()

print(matplotlib.get_backend())

plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)
MY_PURPLE = (107, 76, 154)
MY_LIME = (148, 139, 61)
MY_BRICK = (146, 36, 40)
MY_GRAY = (128, 133, 133)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


font = {'size': 20}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    cols[column].append(float(row[column]))
    return cols


"""
Each dataset is "contained" in a dictionary.
These are the keywords defined for each dictionary that describe the 
dataset.
"""
file_name = "file_name"
xlabel = "xlabel"
legend_labels = "legend_labels"
nr_cols = "nr_cols"
xlim = "xlim"
legend_position = "legend_position"
legend_cols = "legend_cols"  # number of columns in the legend
bbox = "bbox"

# The datasets for the attacks.

round_channel = {file_name: "round_channel",
               xlabel: "# of bits per channel (CD)",
               legend_labels: ["", "LBFGS", "L inf FGSM", "L1 BIM", "L2 C&W"],
               nr_cols: 5,
               xlim: (0, 8),
               legend_position: "lower center",
               legend_cols: 1,
               bbox: (0.0, 0.1)}

fft_channel = {file_name: "fft_channel",
               xlabel: "Compression (%) in the frequency domain (FC)",
               legend_labels: ["", "LBFGS", "L inf FGSM", "L1 BIM", "L2 C&W"],
               nr_cols: 5,
               xlim: (0, 80),
               legend_position: "lower left",
               legend_cols: 2,
               bbox: (0.0, 0.1)}

noise_channel = {file_name: "noise_channel",
               xlabel: "Strength of the Uniform noise (epsilon)",
               legend_labels: ["", "LBFGS", "L inf FGSM", "L1 BIM", "L2 C&W"],
               nr_cols: 5,
               xlim: (0, 0.5),
               legend_position: "lower left",
               legend_cols: 2,
               bbox: (0.0, 0.1)}

gauss_channel = {file_name: "gauss_channel",
               xlabel: "Strength of the Gaussian noise (sigma)",
               legend_labels: ["", "LBFGS", "L inf FGSM", "L1 BIM", "L2 C&W"],
               nr_cols: 5,
               xlim: (0, 0.5),
               legend_position: "lower left",
               legend_cols: 2,
               bbox: (0.0, 0.1)}



# You can easily change where each of the datasets is placed in the
# final grid of the figure.
# datasets = [uniform, gaussian,
#             contrast, ]
# datasets = [uniform, ]
# datasets = [uniform, gaussian, ]
# datasets = [uniform, gaussian, contrast, ]
datasets = [round_channel, fft_channel, gauss_channel, noise_channel]
# datasets = [round_channel]
# datasets = [translations]
# datasets = [gradientSign]

ncols = 1
nrows = int(np.ceil(len(datasets) / ncols))

# figsize: width, height in inches
fig = plt.figure(figsize=(ncols * 9, nrows * 6))
# fig = plt.figure()

colors = [get_color(color) for color in
          ["", MY_GREEN, MY_BLUE, MY_ORANGE, MY_LIME, MY_BLACK,
           MY_PURPLE, MY_GRAY, MY_BRICK, MY_RED, MY_RED]]

markers = ["", "+", "o", "v", "s", "D", "^", "+", "o", "v", "s", "D", "^"]
linestyles = ["", ":", "-", "-.", "--", ":", "-", "-.", "--", ":", "-"]

for j, data in enumerate(datasets):
    print("dataset: ", data)
    plt.subplot(nrows, ncols, j + 1)
    cols = read_columns(data[file_name], columns=data[nr_cols])

    # debug
    print("col 0: ", cols[0])
    print("col 1: ", cols[1])

    for i in range(data[nr_cols]):
        if i > 0:  # skip first column with the epoch number
            plt.plot(cols[0], cols[i], label=f"{data[legend_labels][i]}", lw=3,
                     color=colors[i], linestyle=linestyles[i],
                     marker=markers[i])

    plt.grid()
    plt.legend(loc=data[legend_position],
               ncol=data[legend_cols],
               frameon=False,
               prop={'size': 18},
               # bbox_to_anchor=data[bbox],
               )
    plt.xlabel(data[xlabel])
    # plt.title(titles[j], fontsize=16)
    plt.ylabel("Test accuracy (%)")
    plt.ylim(0, 100)
    plt.xlim(data[xlim][0], data[xlim][1])

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.show(block=True)
plt.interactive(False)
fig.savefig(dir_path + "/" + "attacks8.pdf", bbox_inches='tight')
plt.close()
print("Time (sec): ", time.time() - start)