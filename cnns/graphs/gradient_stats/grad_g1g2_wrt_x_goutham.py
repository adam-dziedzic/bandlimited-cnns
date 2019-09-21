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

values = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-17-47-48-005485_grad_stats_False.csv",
    file_name: "2019-09-20-22-47-14-480927_cifar10_grad_stats.csv",
    title: "eta * gradient vs eta * x",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 60,
    legend_cols: 2,
    labels: ['no labels'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

datasets = [values,
            # not_recovered,
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

    base_col = 12
    print(f'cols[{base_col}][0]: ', cols[base_col][0])
    assert cols[base_col][0] == 'eta_grad'
    print(f'cols[{base_col + 2}][0]: ', cols[base_col + 2][0])
    assert cols[base_col + 2][0] == 'eta_x'

    eta_grad = cols[base_col + 1]
    eta_x = cols[base_col + 3]
    print("col eta_grad: ", eta_grad)
    print("col eta_x: ", eta_x)
    print('col length: ', len(eta_grad))

    plt.plot(eta_x, eta_grad,
             # label=f"{dataset[labels][0]}",
             # lw=line_width,
             color=colors[j],
             # linestyle=linestyles[j],
             linestyle='None',
             marker=markers[j % len(markers)])

    plt.grid()
    # plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
    #            frameon=True,
    #            prop={'size': legend_size},
    #            # bbox_to_anchor=dataset[bbox]
    #            )
plt.ylabel('eta * grad')
plt.xlabel('eta * x')
plt.title('Gradient of (max logit - 2nd max logit) w.r.t. input x',
          fontsize=title_size)

# plt.ylim((0, 20))
# plt.xlim((0, 20))

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "grad_g1g2_wrt_x_goutham_2." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
