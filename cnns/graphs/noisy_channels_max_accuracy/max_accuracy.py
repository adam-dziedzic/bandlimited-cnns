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
legend_size = 30
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
                    cols[column].append(float(row[column]))
    return cols


ylabel = "ylabel"
xlabel = "xlabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
column_nr = "column_nr"
labels = "labels"
legend_cols = "legend_cols"
ylim = "ylim"
xlim = "xlim"



cifar_compress = {ylabel: "Accuracy (%)",
                  xlabel: "Compression rate (%)",
                  file_name: "cifar-compress",
                  title: "accuracy",
                  legend_pos: "upper left",
                  bbox: (0.0, 1.05),
                  column_nr: 4,
                  legend_cols: 3,
                  labels: ["", "FC", 'CD', "SVD"],
                  xlim: (0, 90),
                  ylim: (30, 100),
                  title: "Cifar10"}

cifar_noise = {ylabel: "Accuracy (%)",
               xlabel: "Epsilon (strength of the noise)",
               file_name: "cifar-noisy",
               title: "accuracy",
               legend_pos: "upper left",
               bbox: (0.0, 1.05),
               column_nr: 4,
               legend_cols: 3,
               labels: ["", "Gauss", "Uniform", "Laplace"],
               xlim: (0, 0.1),
               ylim: (47, 100),
               title: "Cifar10"}

imagenet_compress = {ylabel: "Accuracy (%)",
                  xlabel: "Compression rate (%)",
                  file_name: "imagenet-compress",
                  title: "accuracy",
                  legend_pos: "upper left",
                  bbox: (0.0, 1.05),
                  column_nr: 4,
                  legend_cols: 3,
                  labels: ["", "FC", 'CD', "SVD"],
                  xlim: (0, 90),
                  ylim: (0, 90),
                  title: "ImageNet"}

imagenet_noise = {ylabel: "Accuracy (%)",
               xlabel: "Epsilon (strength of the noise)",
               file_name: "imagenet-noisy",
               title: "accuracy",
               legend_pos: "upper left",
               bbox: (0.0, 1.05),
               column_nr: 4,
               legend_cols: 3,
               labels: ["", "Gauss", "Uniform", "Laplace"],
               xlim: (0, 0.1),
               ylim: (55, 80),
               title: "ImageNet"}

colors = [get_color(color) for color in
          ["", MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = ["", "-", "--", ":", "-", "--", ":", "-"]

datasets = [cifar_compress, cifar_noise, imagenet_compress, imagenet_noise]

# width = 12
# height = 5
# lw = 3

width = 30
height = 7
lw = 4

fig = plt.figure(figsize=(width, len(datasets) * height))

for j, dataset in enumerate(datasets):
    plt.subplot(len(datasets) // 2, 2, j + 1)
    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    print("col 0: ", cols[0])
    print("col 1: ", cols[1])

    for i in range(columns):
        if i > 0:  # skip first column with the epoch number
            plt.plot(cols[0], cols[i], label=f"{dataset[labels][i]}", lw=lw,
                     color=colors[i], linestyle=linestyles[i])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
               frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.xlabel(dataset[xlabel])
    plt.title(dataset[title], fontsize=title_size)
    plt.ylabel(dataset[ylabel])
    plt.ylim(dataset[ylim])
    # plt.xscale('log', basex=2)
    plt.xlim(dataset[xlim])

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "noisy_channels_max_accuracy." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
