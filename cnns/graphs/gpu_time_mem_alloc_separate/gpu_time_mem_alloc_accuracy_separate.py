import matplotlib
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


fig = plt.figure(figsize=(8, 6))

titles = ["ResNet-18 on CIFAR-10", "DenseNet-121 on CIFAR-100"]
datasets = ["cifar10", "cifar100"]

rows = 2
cols = 2
iter = 1
subplots = ["compression "]

# files = ["0-fp16", "0-fp32"]
for i, dataset in enumerate(datasets):
    compression, accuracy, time, mem = read_columns(dataset)
    lw = 3
    fontsize = 14

    plt.subplot(rows, cols, iter)
    iter += 1
    label = "GPU mem\nallocated"
    plt.plot(compression, mem, label=label, lw=lw,
             marker="v", color=get_color(MY_RED))
    plt.grid()
    plt.legend(loc='lower left', frameon=False, prop={'size': fontsize})
    plt.xlabel('Compression ratio (%)', fontsize=fontsize)
    plt.title(titles[i], fontsize=fontsize)
    if iter % 2 == 0:
        plt.ylabel("Normalized\nperformance (%)", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(20, 130)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 80)

    plt.subplot(rows, cols, iter)
    iter +=1
    label = "Epoch time"
    plt.plot(compression, time, label=label, lw=lw,
             marker="s", color=get_color(MY_BLUE))
    plt.grid()
    plt.legend(loc='lower left', frameon=False, prop={'size': fontsize})
    plt.xlabel('Compression ratio (%)', fontsize=fontsize)
    plt.title(titles[i], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(20, 130)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 80)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show()
fig.savefig(dir_path + "/" + "gpu_time_mem_alloc_accuracy_separate.pdf",
            bbox_inches='tight')
