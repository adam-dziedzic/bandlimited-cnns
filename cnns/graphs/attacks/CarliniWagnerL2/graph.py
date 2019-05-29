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
contrast = {file_name: "ContrastReductionAttack",
            xlabel: "Strength of contrast based attack\n"
                    "(epsilon)",
            legend_labels: ["", "R=2", "R=4", "R=8", "R=16", "R=32", "R=64",
                            "R=128",
                            "R=256", "full spectra\n(FP32-C=0%)",
                            "band-limited\n(FP32-C=85%)"],
            nr_cols: 11,
            xlim: (0, 1),
            legend_position: "lower left",
            legend_cols: 2,
            bbox: (0.0, 0.1)}

gaussian = {file_name: "cifar10-on-ResNet18-add-gaussian-noise",
            xlabel: "Level of Gaussian noise (sigma)",
            legend_labels: ["sigma", "FP32-C=0%", "FP16-C=0%",
                            "FP32-C=0%\nearly stopping",
                            "FP32-C=50%", "FP32-C=85%"],
            nr_cols: 6,
            xlim: (0, 2),
            legend_position: "upper right",
            legend_cols: 1,
            bbox: (0.0, 0.0)}

uniform = {file_name: "cifar10-on-ResNet18-additive-uniform-noise",
           xlabel: "Level of uniform noise (sigma)",
           legend_labels: ["epsilon", "full spectra\n(FP32-C=0%)",
                           "band-limited\n(FP32-C=85%)"],
           nr_cols: 3,
           xlim: (0, 1),
           legend_position: "upper right",
           legend_cols: 1,
           bbox: (0.0, 0.0)}

multiple = {file_name: "MultiplePixelsAttack",
            xlabel: "Number of perturbed pixels\n"
                    "(for the multiple pixel attack)",
            legend_labels: ["iterations", "full spectra\n(FP32-C=0%)",
                            "band-limited\n(FP32-C=85%)"],
            nr_cols: 3,
            xlim: (0, 1000),
            legend_position: "upper right",  # "upper right",
            legend_cols: 1,
            bbox: (0.0, 0.0)}

local_search = {file_name: "LocalSearchAttack",
                xlabel: "Number of perturbed pixels\n"
                        "(for the local search attack)",
                legend_labels: ["number of perturbed pixels", "full spectra\n(FP32-C=0%)",
                                "band-limited\n(FP32-C=85%)"],
                nr_cols: 3,
                xlim: (0, 21),
                legend_position: "upper right",  # "upper right",
                legend_cols: 1,
                bbox: (0.0, 0.0)}

rotations = {file_name: "rotations",
                xlabel: "Angle of the rotation\n"
                        "(in degrees for the spatial attack)",
                legend_labels: ["Angle of the rotation", "full spectra\n(FP32-C=0%)",
                                "band-limited\n(FP32-C=85%)"],
                nr_cols: 3,
                xlim: (0, 20),
                legend_position: "lower left",  # "upper right",
                legend_cols: 1,
                bbox: (0.0, 0.0)}

translations = {file_name: "translations",
                xlabel: "Limits of the horizontal and vertical translations"
                        "\n(in pixels for the spatial attack)",
                legend_labels: ["horizontal and vertical "
                                "translations\n(limits in pixels)",
                                "full spectra\n(FP32-C=0%)",
                                "band-limited\n(FP32-C=85%)"],
                nr_cols: 3,
                xlim: (0, 21),
                legend_position: "upper right",  # "upper right",
                legend_cols: 1,
                bbox: (0.0, 0.0)}

gradientSign = {file_name: "gradientSignAttack",
                xlabel: "Strength of the gradient sign attack\n"
                        "(epsilon)",
                legend_labels: ["Epsilon",
                                "full spectra\n(FP32-C=0%)",
                                "band-limited\n(FP32-C=85%)"],
                nr_cols: 3,
                xlim: (0.0, 0.2),
                legend_position: "upper right",  # "upper right",
                legend_cols: 1,
                bbox: (0.0, 0.0)}



# You can easily change where each of the datasets is placed in the
# final grid of the figure.
# datasets = [uniform, gaussian,
#             contrast, ]
# datasets = [uniform, ]
# datasets = [uniform, gaussian, ]
# datasets = [uniform, gaussian, contrast, ]
datasets = [uniform, gaussian, contrast, multiple, local_search, rotations,
            translations, gradientSign]
# datasets = [translations]
# datasets = [gradientSign]

ncols = 2
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
fig.savefig(dir_path + "/" + "attacks.pdf", bbox_inches='tight')
print("Time (sec): ", time.time() - start)