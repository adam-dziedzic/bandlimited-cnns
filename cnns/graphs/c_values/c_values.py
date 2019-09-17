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
fontsize = 30
legend_size = 22
title_size = 30
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280

wrt = 'inputs'


# wrt = 'model_parameters'

def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i == 0:
                continue  # omit the header
            for column in range(columns):
                try:
                    # print('column: ', column)
                    cols[column].append(float(row[column]))
                except ValueError as ex:
                    print("Exception: ", ex)
    return cols


def read_rows(dataset, row_nr=None):
    file_name = dir_path + "/" + dataset
    rows = []
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=";", quotechar='|')
        for i, row in enumerate(data):
            result = []
            if rows and i > row_nr:
                break
            for val in row:
                try:
                    # print('column: ', column)
                    result.append(float(val))
                except ValueError as ex:
                    print("Exception: ", ex)
            rows.append(result)
    return rows


ylabel = "ylabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
column_nr = "column_nr"
row_nr = "row_nr"
labels = "labels"
legend_cols = "legend_cols"
xlim = "xlim"
ylim = "ylim"

original = {  # ylabel: "L2 adv",
    # file_name: "../../nnlib/robustness/2019-09-11-21-11-34-525829-len-32-org-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-15-11-046445-len-32-org-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-39-58-229597-len-1-org_recovered-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-15-52-21-873237-len-5-org-images-eigenvals-min-avg-max",
    # file_name: "../../nnlib/robustness/2019-09-12-16-03-19-884343-len-17-org-images-eigenvals-confidence",
    # file_name: "../../nnlib/robustness/2019-09-12-10-28-45-366327-len-62-org-images-highest_eigenvalues",
    file_name: "./c_values.csv",
    title: "",
    legend_pos: "lower left",
    # legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 10,
    row_nr: 13,
    legend_cols: 1,
    labels: ['c values',
             'empty channel',
             'Gauss 0.03',
             'CD 4 bits',
             'FC 16%',
             'Uniform 0.03',
             'SVD 50%',
             'Laplace 0.03',
             'reduce brightness -43 RGB',
             'rand self-ensemble'
             ],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN]]
markers = ["+", "o", "v", "s", "D", "^", "+", 'o', 'v', '+']
linestyles = [":", "-", "--", ":", "-", "--", "-", "--", ':', ':']

datasets = [
    original,
]
dataset = original

# width = 12
# height = 5
# lw = 3
fig_size = 10
width = 18
height = 10
line_width = 4
markersize = 20
layout = "horizontal"  # "horizontal" or "vertical"

fig = plt.figure(figsize=(len(datasets) * width, height))
xlen = 20
indexing = []

cols = read_columns(dataset[file_name], columns=dataset[column_nr])
for j, col in enumerate(cols):
    if j == 0:
        indexing = col
        print('indexing: ', indexing)
    else:
        values = col
        print(f"col {j}: ", col)
        plt.plot(indexing,
                 values,
                 label=dataset[labels][j],
                 lw=line_width,
                 color=colors[j],
                 linestyle=linestyles[j],
                 # linestyle='None',
                 marker=markers[j % len(markers)],
                 markersize=markersize)

plt.grid()
plt.legend(  # loc='upper right',
    loc='lower left',
    ncol=1,
    frameon=False,
    prop={'size': legend_size},
    title='Channel type:',
    # bbox_to_anchor=dataset[bbox]
)
plt.ylabel('Test accuracy (%)')
plt.xlabel('$c$ parameter (in the C&W $L_2$ attack)')
plt.xticks(indexing)
plt.xscale('log', basex=10)
# plt.title(f'Highest eigenvalues of Hessians w.r.t. {wrt} for 62 images',
#           fontsize=title_size)

# plt.ylim((0,20))
# plt.xlim((0, xlen))

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
# plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + f"c_values." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
