import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import os

print(matplotlib.get_backend())

plt.interactive(True)
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
legend_font=16
fontsize=25
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

GPU_MEM_SIZE = 16280
columns = 6


def read_columns(dataset):
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

# width=8
# height=6

width=8
height=6

fig = plt.figure(figsize=(width, height))

dataset = "cifar10-on-ResNet18-add-gaussian-noise"
labels = ["sigma", "FP32-C=0% full spectra",
          "FP16-C=0% full spectra\n(reduced precision: 16 bits)",
          "FP32-C=0% early stopping",
          "FP32-C=50% band-limited",
          "FP32-C=85% band-limited"]

cols = read_columns(dataset)
colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_BLACK, MY_ORANGE, MY_GOLD, MY_RED]]
markers = ["^", "o", "v", "s", "D", "p"]  # ^ - triangle up, o - circle, v - triangle down, D - diamond, p - pentagon;

for i in range(columns):
    if i > 0:  # skip sigma
        plt.plot(cols[0], cols[i], label=labels[i], lw=3, marker=markers[i],
                 color=colors[i])

plt.grid()
plt.legend(loc='upper right', frameon=False, prop={'size': legend_font},
           bbox_to_anchor=(1, 1))
plt.xlabel('Level of Gaussian noise (sigma)')
# plt.title(dataset, fontsize=16)
plt.ylabel("Test accuracy (%)")
plt.ylim(0, 100)
plt.xlim(0, 2)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.show(block=True)
plt.interactive(False)
format="png" # or "pdf"
fig.savefig(dir_path + "/" + "gaussian_noise_font." + format,
            bbox_inches='tight',
            transparent=True)
