import matplotlib

# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os

from cnns.nnlib.utils.general_utils import get_log_time

print(matplotlib.get_backend())

delimiter = ';'
classes_nr = 10

ylabel_size = 25
font = {'size': 30}
matplotlib.rc('font', **font)
lw = 4
fontsize = 25
legend_size = 24
title_size = fontsize
font = {'size': fontsize}
matplotlib.rc('font', **font)

legend_position = 'right'
frameon = False
bbox_to_anchor = (0.0, -0.1)

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


line_width = 4
colors = [get_color(color) for color in
          [MY_RED, MY_BLUE, MY_RED, MY_GREEN, MY_BLACK, MY_GOLD,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN, MY_GREEN, MY_BLACK]]
markers = ["o", "+", "^", "v", "D", "^", "+", 'o', 'v', '+', 'o', '+']
linestyles = ["-", "--", ":", "--", "-", "--", "-", "--", ':', ':', '-', '--']

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_columns(dataset):
    file_name = dir_path + "/" + dataset + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        header = []
        cols = []

        for i, row in enumerate(data):
            # print("i, row: ", i, row)
            for column, value in enumerate(row):
                # get rid of any feed line characters or any other weired
                # white characters
                value = str(value).strip()
                value = value.replace('\r', '')
                value = value.replace('\n', '')
                value = value.replace(chr(194), '')
                value = value.replace(chr(160), '')
                if i > 0:  # skip header
                    if len(cols) < column + 1:
                        cols.append([])
                    cols[column].append(float(value))
                else:
                    header.append(value)

    return header, cols


# width=8
# height=6

width = 10
height = 7.5

fig = plt.figure(figsize=(width, height))

# dataset = "model_perturb_data5"
dataset = "data1"
# dataset = "model_perturb_data_roubst+param4"
# dataset = "perturb_conv2"
labels, cols = read_columns(dataset)

for i, column_values in enumerate(cols):
    if i > 0:  # skip sigma
        plt.plot(cols[0], column_values, label=labels[i], lw=3,
                 marker=markers[i],
                 color=colors[i])

plt.grid()
# plt.legend(loc='lower left', frameon=False, prop={'size': legend_size},
#            # bbox_to_anchor=(1, 1),
#            ncol=2)
plt.xlabel('FFT compression rate (%)')
# plt.title(dataset, fontsize=16)
plt.ylabel("Test accuracy (%)")
# plt.ylim(0, 100)
# plt.xlim(0, 2)
# plt.xscale('log', basex=10)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.show(block=True)
plt.interactive(False)
format = ".pdf"  # ".png" or ".pdf"
fig.savefig(dir_path + "/" + dataset + get_log_time() + format,
            bbox_inches='tight',
            transparent=True)
plt.close()
