import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE=(56,106,177)
MY_RED=(204,37,41)
MY_ORANGE=(218,124,48)
MY_GREEN=(62,150,81)
MY_BLACK=(83,81,84)

def get_color(COLOR_TUPLE_255):
    return [x/255 for x in COLOR_TUPLE_255]

# metric = "GPUUtilization"
metric = "MemoryUsed"
if metric == "MemoryUtilization":
    prefix = "MemoryUtilization"
    ylabel= 'Memory \n utilization (%)'
elif metric == "GPUUtilization":
    prefix = "GPUUtilization"
    ylabel='GPU \n utilization (%)'
elif metric == "MemoryUsed":
    prefix = "MemoryUsed"
    ylabel = 'Memory \n used (%)'

font = {'size': 20}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

max_length = 20000
GPU_MEM_SIZE = 16280

# Data dir prefix:

# dir_prefix = "mem_test_data_utilization"
# dir_prefix = "mem_test_data_utilization_train_test"
# dir_prefix = "mem_test_data_utilization_train_test_2_epochs"
# dir_prefix = "no_mem_test_data_utilization_train_test_2_epochs"
# dir_prefix = "no_mem_test_data_utilization_train_test_3_iterations"
# dir_prefix = "mem_test_mem_used_train_test_3_iterations"
# dir_prefix = "mem_test_mem_used_train_test_3_iterations_only_main-fp16"
dir_prefix = "mem_test_mem_used_train_test_3_iterations_only_main_fp16_correct_compress_rate"

def read_columns(rate):
    file_name = dir_path + "/" + dir_prefix + "/utilization-" + str(
        rate) + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        column1, column2, column3 = [], [], []

        for i, row in enumerate(data):
            if i > 0:
                # print(row[1][:-1])
                column1.append(int(row[0][:-1]))
                column2.append(int(row[1][:-1]))
                column3.append(int(row[2][:-3])/GPU_MEM_SIZE*100)
    column1 = np.pad(column1, (0, max_length - len(column1)), 'constant')
    column2 = np.pad(column2, (0, max_length - len(column2)), 'constant')
    column3 = np.pad(column3, (0, max_length - len(column2)), 'constant')
    min = 1000 # 3000
    max = 8000
    if rate == "fp16-0":
        min = 1000
        max = 8000
    column1, column2, column3 = column1[min:max], column2[min:max], column3[min:max]
    print(file_name)
    if metric == "MemoryUtilization":
        column = column2
    elif metric == "GPUUtilization":
        column = column1
    elif metric == "MemoryUsed":
        column = column3
    else:
        raise Exception(f"Unknown metric: {metric}")

    print("avg:", np.average(column), "max:", np.max(column), "median:",
          np.median(column))
    return column1, column2, column3


fig = plt.figure(figsize=(8, 6))

# files = ["0-fp16", "0-fp32", "25-fp32", "50-fp32", "75-fp32"]
files = [["fp32-0", "fp32-50", "fp32-75"], ["fp32-0", "fp16-0", "fp16-50"]]

# files = ["0-fp16", "0-fp32"]
for i, file_nr in enumerate(files):
    plt.subplot(2, 1, i + 1)
    for j, rate in enumerate(file_nr):
        if rate == "fp32-0":
            # color = "blue"
            color = MY_BLUE
        elif rate == "fp16-0":
            # color = "green"
            color = MY_GREEN
        elif rate == "fp32-50":
            # color = "red"
            color = MY_RED
        elif rate == "fp32-75":
            # color = "black"
            color = MY_BLACK
        else:
            color = MY_ORANGE
        color = get_color(color)
        gpu, mem, mem_used = read_columns(rate)
        label = str(rate) + "%"
        if metric == "MemoryUtilization":
            plt.plot([x for x in range(len(mem))], mem, label=label, lw=3,
                     marker=i, color=color)
        elif metric == "GPUUtilization":
            plt.plot([x for x in range(len(gpu))], gpu, label=label, lw=3,
                     marker=i, color=color)
        elif metric == "MemoryUsed":
            plt.plot([x for x in range(len(mem_used))], mem_used, label=label, lw=3,
                     marker=i, color=color)
        else:
            raise Exception(f"Unknown metric: {metric}")

    plt.grid()
    plt.legend(loc='upper left', frameon=False, ncol=len(file_nr),
               prop={'size': 16}, bbox_to_anchor=(0, 1.08))
    if i > 0:
        plt.xlabel('Time (msec)')
        plt.title("FFT-based and mixed-precision compressions", fontsize=16)
    else:
        plt.title("FFT-based compression (only)", fontsize=16)
    plt.ylabel(ylabel)
    plt.ylim(0, 20)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show()
fig.savefig(dir_path + "/" + dir_prefix + "/" + prefix + "-utilization.pdf",
            bbox_inches='tight')
