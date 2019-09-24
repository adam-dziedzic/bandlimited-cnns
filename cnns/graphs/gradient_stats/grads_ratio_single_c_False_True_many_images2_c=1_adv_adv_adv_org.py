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

recovered = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['recovered'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 56,
    legend_cols: 2,
    labels: ['not recovered'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

datasets = [recovered,
            not_recovered,
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
# limit = 4096

fig = plt.figure(figsize=(len(datasets) * width, height))

for j, dataset in enumerate(datasets):
    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    org_col = 30
    adv_col = 28
    print(f'cols[{org_col}][0]: ', cols[org_col][0])
    assert cols[org_col][0] == 'l2_norm_adv_correct'
    print(f'cols[{adv_col}][0]: ', cols[adv_col][0])
    assert cols[adv_col][0] == 'l2_norm_adv_adv'

    org_grad = cols[org_col + 1]
    adv_grad = cols[adv_col + 1]
    print("col org: ", org_grad)
    print("col adv: ", adv_grad)
    print('col length: ', len(adv_grad))

    plt.plot(adv_grad, org_grad, label=f"{dataset[labels][0]}",
             # lw=line_width,
             color=colors[j],
             # linestyle=linestyles[j],
             linestyle='None',
             marker=markers[j % len(markers)])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
               frameon=True,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
plt.ylabel('$L_2$ of gradient for adv image and org class')
plt.xlabel('$L_2$ of gradient for adv image and adv class')
# plt.title('Gradients ratio and recovery', fontsize=title_size)

plt.xlim((1e-5, 1e+2))
plt.ylim((1e-1, 1e+2))
plt.xscale('log', basex=10)
plt.yscale('log', basey=10)
# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "grads_ratio_single_c_False_True_many_images2_c=1_ver7." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
