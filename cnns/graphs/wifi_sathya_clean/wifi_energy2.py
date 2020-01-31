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
MY_VIOLET = (107, 76, 154)
MY_BROWN = (146, 36, 40)
MY_OWN = (25, 150, 10)

def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# fontsize=20
fontsize = 30
legend_size = 24
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
            if i > 0:  # skip header
                for column in range(columns):
                    value = row[column]
                    value = str(value).strip()
                    value = value.replace('\r', '')
                    value = value.replace('\n', '')
                    value = value.replace(chr(194), '')
                    value = value.replace(chr(160), '')
                    print('value: ', value)
                    cols[column].append(float(value))
    return cols


ylabel = "ylabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
labels = "labels"

energy2 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi_energy2",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "2 Wi-Fis"],
}

energy3 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi_energy3",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "3 Wi-Fis"],
}

energy4 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi_energy4",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "4 Wi-Fis"],
}

energy5 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi_energy5",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "5 Wi-Fis"],
}

energyAll2 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi_data2",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "2 Wi-Fis",
    "3 Wi-Fi", "4 Wi-Fi", "5 Wi-Fis"],
}

wifi2 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi2",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "2 Wi-Fis"],
}

wifi3 = {
    ylabel: "Energy (dBm)",
    file_name: "wifi3",
    title: "accuracy",
    legend_pos: "upper left",
    bbox: (0.0, 0.1),
    labels: ["", "0 Wi-Fi", "1 Wi-Fi", "2 Wi-Fis", "3 Wi-Fis"],
}

ncols = [4, 3]
columns = 5

colors = [get_color(color) for color in
          ["", MY_GREEN, MY_BLUE, MY_ORANGE, MY_BROWN, MY_BLACK,
           MY_GOLD, MY_VIOLET, MY_RED, MY_OWN]]
markers = ["+", "o", "v", "s", "D", "^", "+", "o", "v", "s", "D", "^"]
linestyles = ["", "-", "--", ":", "--", "--", ":", "-", "--", ":"]

datasets = [wifi3]
# datasets = [energy2]

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
    cols = read_columns(dataset[file_name], columns=columns)

    # print("col 0: ", cols[0])
    # print("col 1: ", cols[1])

    for i in range(columns):
        print("i: ", i)
        if i > 1:  # skip first column with the index number
            plt.plot(cols[0], cols[i], label=f"{dataset[labels][i]}", lw=lw,
                     color=colors[i], linestyle=linestyles[i])

    plt.grid()
    plt.legend(loc=dataset[legend_pos], ncol=ncols[j], frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.xlabel('Sample number')
    # plt.title(titles[j], fontsize=16)
    plt.ylabel(dataset[ylabel])
    plt.ylim(-40, -15)
    plt.xlim(0, 2048)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = ".pdf"  # "pdf" or "png"
destination = dir_path + "/" + dataset[file_name] + '-graph2' + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            transparent=True,
            )
plt.show(block=True)
# plt.interactive(False)
plt.close()
