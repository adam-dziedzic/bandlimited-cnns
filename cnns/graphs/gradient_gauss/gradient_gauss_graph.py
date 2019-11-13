import matplotlib
import numpy as np

# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os

from cnns.nnlib.utils.general_utils import get_log_time

print(matplotlib.get_backend())

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (57, 106, 177)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_RED = (204, 37, 41)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)
MY_VIOLET = (107, 76, 154)
MY_BROWN = (146, 36, 40)
MY_OWN = (25, 150, 10)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# fontsize=20
fontsize = 25
legend_size = fontsize
title_size = fontsize
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280

delimiter=','


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    try:
                        # print('column: ', column)
                        cols[column].append(float(row[column]))
                    except ValueError as ex:
                        pass
                        cols[column].append(row[column])
                        # print("Exception: ", ex)
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
image = "image"
adv = "adv"
org = "original"
gauss = "gauss"

gradient_gauss_data = {
    file_name: 'gradient_gauss_data.csv',
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "center left",
    # bbox: (0.0, 0.0),
    column_nr: 4,
    legend_cols: 1,
    labels: ['gauss image', 'original image (baseline)', 'adv image (FGSM)'],
    xlim: (0, 0.3),
    ylim: (0, 4)}


colors = [get_color(color) for color in
          [MY_BLACK, MY_BLUE, MY_RED, MY_BLUE, MY_BLACK, MY_GOLD,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN]]
markers = ["o", "", "", "v", "D", "^", "+", 'o', 'v', '+']
linestyles = ["-", "--", "--", "--", "-", "--", "-", "--", ':', ':']

datasets = [
    gradient_gauss_data,
]

# width = 12
# height = 5
# lw = 3
decimals = 3
fig_size = 10
width = 12
height = 6
line_width = 4
layout = "horizontal"  # "horizontal" or "vertical"
# limit = 4096
nrows = 1
ncols = 1

fig = plt.figure(figsize=(ncols * width, nrows * height))
dist_recovered = None

for j, dataset in enumerate(datasets):

    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    x_data = cols[0]
    plt.grid()
    dataset_labels = dataset[labels]
    for i, col in enumerate(cols[1:]):
        plt.plot(x_data, col,
                 label=dataset_labels[i % len(dataset_labels)],
                 lw=line_width,
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 # linestyle='None',
                 marker=markers[i % len(markers)])

    plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
               frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.ylabel('$E_{x}||\partial_{x}\mathcal{L}||_2$')
    plt.xlabel('$\sigma$ (strength of the Gaussian noise)')
    # plt.title('The original image')

    plt.xlim(dataset[xlim])
    plt.ylim(dataset[ylim])
    # plt.xscale('log', basex=10)
    # plt.yscale('log', basey=10)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "gauss_grads_" + get_log_time() + '.' + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
