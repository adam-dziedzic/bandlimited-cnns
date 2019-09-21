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
legend_size = 22
title_size = 30
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=";", quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    try:
                        print('column: ', column)
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

no_259 = {  # ylabel: "L2 adv",
    file_name: "2019-09-09-17-47-48-005485_grad_stats_False.csv",
    title: "no_259",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['no_259'],
    xlim: (0, 100),
    ylim: (0, 100)}

no_200 = {  # ylabel: "L2 adv",
    file_name: "2019-09-10-12-01-04-067880_grad_stats_False.csv",
    title: "no_200",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['no_200'],
    xlim: (0, 100),
    ylim: (0, 100)}

no_1070 = {  # ylabel: "L2 adv",
    file_name: "2019-09-09-19-00-34-169703_grad_stats_False.csv",
    title: "no_1070",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['no_1070'],
    xlim: (0, 100),
    ylim: (0, 100)}

yes_1070 = {  # ylabel: "L2 adv",
    file_name: "2019-09-09-19-00-34-169703_grad_stats_True.csv",
    title: "yes_1070",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['yes_1070'],
    xlim: (0, 100),
    ylim: (0, 100)}



yes_100 = {  # ylabel: "L2 adv",
    file_name: "2019-09-10-12-00-06-403683_grad_stats_True.csv",
    title: "yes_100",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['yes_100'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

datasets = [no_259,
            no_200,
            yes_100,
            # yes_1070,
            # no_1070,
            # carlini_imagenet,
            # pgd_cifar10,
            # random_pgd_cifar10,
            # pgd_imagenet,
            # fgsm_imagenet,
            ]

# width = 12
# height = 5
# lw = 3
fig_size = 10
width = 10
height = 10
line_width = 4
layout = "horizontal"  # "horizontal" or "vertical"

fig = plt.figure(figsize=(len(datasets) * width, height))

for j, dataset in enumerate(datasets):
    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    print("col 27: ", cols[27])
    print("col 25: ", cols[25])

    plt.plot(cols[25], cols[27], label=f"{dataset[labels][0]}",
             # lw=line_width,
             color=colors[j], linestyle=linestyles[j],
             marker=markers[j % len(markers)])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
               frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
plt.ylabel('grad L2 for adv class')
plt.xlabel('grad L2 for correct class')
plt.title('Gradients ratio and recovery', fontsize=title_size)

# plt.ylim(dataset[ylim])
# plt.xlim(dataset[xlim])

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "grad_stats_graph." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()