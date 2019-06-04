import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


fontsize = 34  # 20
legend_font = 34
lw = 8  # 3

font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

GPU_MEM_SIZE = 16280


def read_columns(dataset):
    file_name = dir_path + "/" + dataset + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        compression, accuracy, time, mem = [], [], [], []

        for i, row in enumerate(data):
            if i == 1:
                max_accuracy = float(row[2])
                max_epoch_time = float(row[3])
                max_mem_size = float(row[4])
            if i > 0:
                print(row[1])
                compression.append(float(row[1]))
                accuracy.append(float(row[2]) / max_accuracy * 100)
                time.append(float(row[3]) / max_epoch_time * 100)
                mem.append(int(row[4]) / max_mem_size * 100)
    return compression, accuracy, time, mem


fig = plt.figure(figsize=(15, 5.5))

dataset_name = "dataset_name"
title = "title"

cifar10 = {dataset_name: "cifar10",
           title: "ResNet-18 on CIFAR-10"}

cifar100 = {dataset_name: "cifar100",
            title: "DenseNet-121 on CIFAR-100"}

datasets = [cifar10]  # [cifar10, cifar100]

# files = ["0-fp16", "0-fp32"]
for i, dataset in enumerate(datasets):
    plt.subplot(len(datasets), 1, i + 1)
    compression, accuracy, time, mem = read_columns(dataset[dataset_name])
    plt.plot(compression, accuracy, label="Test accuracy", lw=lw,
             marker="o", color=get_color(MY_ORANGE), linestyle="-")
    plt.plot(compression, time, label="Epoch time", lw=lw,
             marker="s", color=get_color(MY_BLUE), linestyle="-.")
    plt.plot(compression, mem, label="GPU mem allocated", lw=lw,
             marker="v", color=get_color(MY_GREEN), linestyle=":")
    plt.grid()
    plt.legend(loc='upper left', frameon=False, prop={'size': legend_font},
               bbox_to_anchor=(0, 0.6))
    if i > -1:
        plt.xlabel('Compression rate (%)')
    plt.title(dataset[title], fontsize=fontsize)
    plt.ylabel("Normalized\n performance (%)")
    # plt.ylim(20, 130)
    # plt.xlim(0, 80)
    plt.ylim(0, 100)
    plt.xlim(0, 86)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show()
format = "png"  # "pdf" or "png"
fig.savefig(
    dir_path + "/images/" + "gpu_time_mem_alloc_accuracy_font2." + format,
    bbox_inches='tight', transparent=True)
plt.close()
