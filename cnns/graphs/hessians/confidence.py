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

wrt = 'inputs'
# wrt = 'model_parameters'

def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=";", quotechar='|')
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
            if i == 0:
                continue  # omit the header
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
    file_name: "../../nnlib/robustness/2019-09-12-16-03-19-884343-len-17-org-images-eigenvals-confidence",
    title: "original",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 3,
    row_nr: 6,
    legend_cols: 3,
    labels: ['original'],
    xlim: (0, 100),
    ylim: (0, 100)}

adversarial = {  # ylabel: "L2 adv",
    # file_name: "../../nnlib/robustness/2019-09-11-21-11-34-525829-len-32-adv-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-15-11-040897-len-32-adv-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-39-58-229330-len-1-adv_recovered-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-15-52-21-871557-len-5-adv-images-eigenvals-min-avg-max",
    file_name: "../../nnlib/robustness/2019-09-12-16-03-19-881953-len-17-adv-images-eigenvals-confidence",
    title: "adversarial",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 3,
    row_nr: 6,
    legend_cols: 3,
    labels: ['adversarial'],
    xlim: (0, 100),
    ylim: (0, 100)}

gauss = {  # ylabel: "L2 adv",
    # file_name: "../../nnlib/robustness/2019-09-11-21-11-34-525829-len-32-adv-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-15-11-050891-len-32-gauss-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-09-39-58-229856-len-1-gauss_recovered-images-eigenvals",
    # file_name: "../../nnlib/robustness/2019-09-12-15-52-21-874077-len-5-gauss-images-eigenvals-min-avg-max",
    # file_name: "../../nnlib/robustness/2019-09-12-16-03-19-886454-len-17-gauss-images-eigenvals-confidence",
    title: "gauss",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 3,
    row_nr: 6,
    legend_cols: 3,
    labels: ['gauss'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

datasets = [
    # original,
    adversarial,
    # gauss,
]

# width = 12
# height = 5
# lw = 3
fig_size = 10
width = 15
height = 10
line_width = 4
markersize = 20
layout = "horizontal"  # "horizontal" or "vertical"

fig = plt.figure(figsize=(len(datasets) * width, height))
xlen = 20
indexing = []

for j, dataset in enumerate(datasets):
    print("dataset: ", dataset)
    row_nr = dataset[row_nr]
    rows = read_rows(dataset[file_name], row_nr=row_nr)
    for i, row in enumerate(rows):
        print(f"row {i}: ", row)
        confidence = row[0]
        eigenvalues = row[1:]
        xlen = len(eigenvalues)
        indexing = [i + 1 for i in range(xlen)]
        plt.plot(indexing,
                 eigenvalues,
                 label=str(confidence),
                 lw=line_width,
                 color=colors[i],
                 linestyle=linestyles[i],
                 # linestyle='None',
                 marker=markers[i % len(markers)],
                 markersize=markersize)

plt.grid()
plt.legend(loc='upper right',
           ncol=1,
           frameon=True,
           prop={'size': legend_size},
           title='Confidence levels:',
           # bbox_to_anchor=dataset[bbox]
           )
plt.ylabel('eigenvalue (value)')
plt.xlabel('i-th eigenvalue')
plt.xticks(indexing)
plt.yscale('symlog', basey=10)
plt.title(f'Spectrum of Hessians w.r.t. {wrt}', fontsize=title_size)

# plt.ylim((0,20))
# plt.xlim((0, xlen))

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + f"eigenvals_confidence_adv_{wrt}9_log." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
