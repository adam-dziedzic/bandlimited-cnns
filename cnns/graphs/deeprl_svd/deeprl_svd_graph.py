"""
How to generate the data?

Go to cnns/deeprl and copy models into dagger_models/env_name_models
Update the return values of the models in the files of the models and
in the script: cnns/deeprl/svd_read_save_model_weights.py. Run the last
script. Update the return values in this script and run it.

"""


import matplotlib

# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

print(matplotlib.get_backend())

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)


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
            for column in range(columns):
                cols[column].append(float(row[column]))
    return cols


ylabel = "ylabel"
returns = 'returns'
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
layer_nr = 'layer_nr'

# return_values = [-12.7, -10.5, -9.6, -8.7, -7.4, -6.2, -4.8]
# return_values = [1, 100, 200, 500, 1000, 10000]
return_values = [10, 100, 1000, 1500, 2000]
columns = len(return_values)

layer_0 = {ylabel: "Singular value (absolute)",
           returns: return_values,
           title: "Magnitude (singular value)",
           legend_pos: "upper right",
           bbox: (0.0, 0.1),
           layer_nr: 0}

layer_1 = {ylabel: "Singular value (absolute)",
           returns: return_values,
           title: "Magnitude (singular value)",
           legend_pos: "upper right",
           bbox: (0.0, 0.1),
           layer_nr: 1}

layer_2 = {ylabel: "Singular value (absolute)",
           returns: return_values,
           title: "Magnitude (singular value)",
           legend_pos: "upper right",
           bbox: (0.0, 0.1),
           layer_nr: 2}

ncols_legend = 1

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_RED, MY_GREEN]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = ["-", "-", "--", ":", ":", '--', '--']

datasets = [layer_0, layer_1, layer_2]

# width = 12
# height = 5
# lw = 3

width = 15
height = 7
lw = 4

fig = plt.figure(figsize=(width, len(datasets) * height))

for j, dataset in enumerate(datasets):
    plt.subplot(len(datasets), 1, j + 1)
    print("dataset: ", dataset)

    cols = []
    for col in range(columns):
        # file_name = '../../deeprl/svd/return_' + str(
        #     dataset[returns][col]) + '.model-' + str(dataset[layer_nr])
        # file_name = '../../deeprl/svd/saved-model-reacher-v2-' + str(dataset[returns][col]) + '-rolls.model-' + str(dataset[layer_nr])
        file_name = '../../deeprl/svd/Ant-v2-' + str(dataset[returns][col]) + '.model-' + str(dataset[layer_nr])
        column = read_columns(dataset=file_name, columns=1)[0]
        # square_col = np.square(column)
        # sum_col = np.sum(square_col)
        # cols.append(square_col / sum_col)
        cols.append(column / np.sum(column))
        # cols.append(column)

    for col_i in range(columns):
        print("return value ", return_values[col_i], ": ", sum(cols[col_i]))

    for i in range(columns):
        plt.plot(range(len(cols[i])), cols[i],
                 label=f"episodes={dataset[returns][i]}",
                 lw=lw, color=colors[i], linestyle=linestyles[i])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=ncols_legend, frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.xlabel('Index (sorted singular values)')
    plt.title(f"SVD for W{dataset[layer_nr]}", fontsize=title_size)
    plt.ylabel(dataset[ylabel])
    # plt.ylim(-40, -15)
    # plt.xlim(0, 10)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.5)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "deeprl_svd_Ant-v2-behavior-clonning_abs_norm." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
plt.show(block=True)
plt.interactive(True)
plt.close()
