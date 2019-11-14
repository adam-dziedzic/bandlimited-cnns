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
fontsize = 43
legend_size = 30
title_size = fontsize
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    try:
                        print('column: ', column)
                        print('row: ', row[column])
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

recovered_0_001 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-01-43-45-115755_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.001, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_RED, MY_GREEN, MY_BLUE, MY_RED,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN]]
markers = ["v", "v", "v", "o", "o", "o", "+", 'o', 'v', '+']
linestyles = ["-", "-", "-", ":", ":", ":", "-", "--", ':', ':']

datasets = [
    'cifar_10_compress_clean.csv',
    'cifar_10_compress_cw.csv',
    'cifar_10_compress_clean.csv',
    'cifar_10_compress_pgd.csv',
    'cifar_10_noise_clean.csv',
    'cifar_10_noise_cw.csv',
    'cifar_10_noise_clean.csv',
    'cifar_10_noise_pgd.csv',
    'imagenet_compress_clean.csv',
    'imagenet_compress_cw.csv',
    'imagenet_noise_clean.csv',
    'imagenet_noise_cw.csv'
]

# width = 12
# height = 5
# lw = 3
decimals = 3
fig_size = 10
width = 15
height = 15
line_width = 5
layout = "horizontal"  # "horizontal" or "vertical"
# limit = 4096
ncols = 2
nrows = len(datasets) // 2 // ncols
assert len(datasets) % 2 == 0

fig = plt.figure(figsize=(ncols * width, nrows * height))
dist_recovered = None

for j, dataset in enumerate(datasets):

    print("dataset: ", dataset)
    columns = 6
    cols = read_columns(dataset, columns=columns)

    if 'cifar' in dataset:
        title = 'CIFAR-10 '
    else:
        title = 'ImageNet '

    if 'compress' in dataset:
        xlim = (0, 100)
        labels = ['FC', 'CD', 'SVD']
        xlabel = 'Compression (%)'
    elif 'noise' in dataset:
        xlim = (0, 0.1)
        labels = ['Gauss', 'Unif', 'Laplace']
        xlabel = '$\epsilon$ (noise strength)'
    else:
        raise Exception('dataset does not contain required words: ' + dataset)

    if 'clean' in dataset:
        suffix = 'Clean'
    elif 'cw' in dataset:
        title += 'C&W'
        suffix = 'C&W'
    elif 'pgd' in dataset:
        suffix = 'PGD'
        title += 'PGD'
    else:
        raise Exception('dataset does not contain required words: ' + dataset)

    labels = [label + " " + suffix for label in labels]

    if j % 2 == 0:
        plt.subplot(nrows, ncols, j // 2 + 1)
        index = -1
        if j % 4 == 0:
            plt.ylabel('Accuracy (%)')

    for i in range(0, len(cols), 2):
        index += 1
        label = labels[index % 3]
        if 'CD' in label or 'Laplace' in label:
            continue
        # print('i: ', cols[i])
        # print('i + 1: ', cols[i + 1])
        plt.plot(cols[i], cols[i + 1],
                 label=label,
                 lw=line_width,
                 color=colors[index % len(colors)],
                 linestyle=linestyles[index % len(linestyles)],
                 # linestyle='None',
                 marker=markers[index % len(markers)],
                 markersize=10)

    if j % 2 == 1:
        plt.grid()
        plt.legend(loc='lower-left', ncol=2,
                   frameon=False,
                   prop={'size': legend_size},
                   markerscale=1,
                   # bbox_to_anchor=dataset[bbox]
                   )
        plt.xlabel(xlabel)
        plt.ylim((0, 100))
        plt.xlim(xlim)
        plt.title(title)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "clean_accuracy_vs_attack_recovery_" + get_log_time() + '.' + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
