import matplotlib

# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os

print(matplotlib.get_backend())

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# fontsize=20
fontsize = 30
legend_size = 26
title_size = 30
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280


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
                    try:
                        cols[column].append(float(row[column]))
                    except ValueError as ex:
                        print("Exception: ", ex)
    return cols


ylabel = "ylabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
column_nr = "column_nr"
labels = "labels"
legend_cols = "legend_cols"
xlim = "xlim"
ylim = "ylim"

many_trials_cifar10 = {ylabel: "Accuracy (%)",
                       file_name: "distortion",
                       title: "CIFAR-10",
                       legend_pos: "upper right",
                       # bbox: (0.0, 0.0),
                       column_nr: 12,
                       legend_cols: 2,
                       labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                       xlim: (0, 12),
                       ylim: (0, 100)}

many_trials_imagenet = {ylabel: "Accuracy (%)",
                        file_name: "distortionImageNet",
                        title: "ImageNet",
                        # legend_pos: "lower left",
                        legend_pos: "upper right",
                        # bbox: (0.0, 0.0),
                        column_nr: 12,
                        legend_cols: 2,
                        labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                        xlim: (0, 100),
                        ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

datasets = [many_trials_cifar10, many_trials_imagenet]

# width = 12
# height = 5
# lw = 3

width = 30
height = 5
line_width = 4
layout = "horizontal"  # "horizontal" or "vertical"

fig = plt.figure(figsize=(width, len(datasets) * height))

for j, dataset in enumerate(datasets):
    if layout == "vertical":
        plt.subplot(len(datasets), 1, j + 1)
    else:
        plt.subplot(1, len(datasets), j + 1)
    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    print("col 0: ", cols[0])
    print("col 1: ", cols[1])

    i = -1
    for col in range(0, columns, 2):
        i += 1
        plt.plot(cols[col], cols[col + 1], label=f"{dataset[labels][i]}", lw=line_width,
                 color=colors[i], linestyle=linestyles[i])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
               frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.xlabel('L2 distortion')
    plt.title(dataset[title], fontsize=title_size)
    plt.ylabel(dataset[ylabel])
    plt.ylim(dataset[ylim])
    plt.xlim(dataset[xlim])

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "distortion2." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
