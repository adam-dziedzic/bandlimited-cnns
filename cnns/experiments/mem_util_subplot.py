import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

is_mem = True
if is_mem:
    prefix = "mem"
else:
    prefix = "GPU"

font = {'size': 20}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

max_length = 10000
# dir_prefix = "mem_test_data_utilization"
# dir_prefix = "mem_test_data_utilization_train_test"
dir_prefix = "mem_test_data_utilization_train_test_2_epochs"

def read_columns(rate):
    file_name = dir_path + "/" + dir_prefix + "/utilization-" + str(
        rate) + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        column1, column2 = [], []

        for i, row in enumerate(data):
            if i > 0:
                # print(row[1][:-1])
                column1.append(int(row[0][:-1]))
                column2.append(int(row[1][:-1]))
    column1 = np.pad(column1, (0, max_length - len(column1)), 'constant')
    column2 = np.pad(column2, (0, max_length - len(column2)), 'constant')
    min = 2500 # 3000
    max = 8000
    if rate == "fp16-0":
        min = 2500
        max = 8000
    column1, column2 = column1[min:max], column2[min:max]
    print(file_name)
    if is_mem:
        column = column2
    else:
        column = column1

    print("avg:", np.average(column), "max:", np.max(column), "median:",
          np.median(column))
    return column1, column2


fig = plt.figure(figsize=(8, 6))

# files = ["0-fp16", "0-fp32", "25-fp32", "50-fp32", "75-fp32"]
files = [["fp32-0", "fp16-0"], ["fp32-0", "fp32-50", "fp32-75"]]

# files = ["0-fp16", "0-fp32"]
for i, file_nr in enumerate(files):
    plt.subplot(2, 1, i + 1)
    for j, rate in enumerate(file_nr):
        if rate == "fp32-0":
            color = "blue"
        elif rate == "fp16-0":
            color = "green"
        elif rate == "fp32-50":
            color = "red"
        elif rate == "fp32-75":
            color = "black"
        else:
            color = "yellow"
        gpu, mem = read_columns(rate)
        label = str(rate) + "%"
        if is_mem:
            plt.plot([x for x in range(len(mem))], mem, label=label, lw=3,
                     marker=i, color=color)
        else:
            plt.plot([x for x in range(len(gpu))], gpu, label=label, lw=3,
                     marker=i, color=color)

    plt.grid()
    plt.legend(loc='upper left', frameon=False)
    if i > 0:
        plt.xlabel('Time (msec)')
    plt.ylabel(prefix + ' util (%)')
    plt.ylim(0, 100)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show()
fig.savefig(dir_path + "/" + dir_prefix + "/" + prefix + "-utilization.pdf",
            bbox_inches='tight')
