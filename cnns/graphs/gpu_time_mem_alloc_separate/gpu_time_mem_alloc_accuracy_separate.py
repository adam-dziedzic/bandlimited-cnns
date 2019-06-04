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


font = {'size': 20}
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
                print("row[2]: ", row[2])
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

# width=10
# height=8
# fontsize=14
# ylim_max=120
# ylim_min=0

width=15
height=4.5
fontsize=30
ylim_max=100
ylim_min=0

fig = plt.figure(figsize=(width, height))


file_name = "file_name"
title = "title"

cifar10 = {file_name: "cifar10",
           title: "ResNet-18 on CIFAR-10"}

cifar100 = {file_name: "cifar100",
            title: "DenseNet-121 on CIFAR-100"}

imagenet = {file_name: "imagenet",
            title: "ResNet-50 on ImageNet"}

datasets = [cifar10] #[cifar10, cifar100, imagenet]

rows = len(datasets)
cols = 2
iter = 1
subplots = ["compression "]

# files = ["0-fp16", "0-fp32"]
for i, dataset in enumerate(datasets):
    print("dataset: ", dataset)
    compression, accuracy, time, mem = read_columns(dataset[file_name])
    lw = 3
    fontsize = fontsize

    plt.subplot(rows, cols, iter)
    iter += 1
    label = "GPU mem\nallocated"
    plt.plot(compression, mem, label=label, lw=lw,
             marker="v", color=get_color(MY_RED))
    plt.grid()
    plt.legend(loc='lower left', frameon=False, prop={'size': fontsize})
    plt.xlabel('Compression rate (%)', fontsize=fontsize)
    plt.title(dataset[title], fontsize=fontsize)
    if iter % 2 == 0:
        plt.ylabel("Normalized\nperformance (%)", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(ylim_min, ylim_max)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 80)

    plt.subplot(rows, cols, iter)
    iter += 1
    label = "Epoch time"
    plt.plot(compression, time, label=label, lw=lw,
             marker="s", color=get_color(MY_BLUE))
    plt.grid()
    plt.legend(loc='lower left', frameon=False, prop={'size': fontsize})
    plt.xlabel('Compression rate (%)', fontsize=fontsize)
    plt.title(dataset[title], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(ylim_min, ylim_max)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 80)

plt.subplots_adjust(hspace=0.6)
# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show(block=True)
format = "png"
fig.savefig(dir_path + "/" + "gpu_time_mem_alloc_font3." + format,
            bbox_inches='tight', transparent=True)
plt.close()