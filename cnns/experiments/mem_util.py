import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

is_mem = False
font = {'size': 20}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

max_length=6700

def read_columns(file_name):
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        column1, column2 = [], []

        for i, row in enumerate(data):
            if i > 0:
                # print(row[1][:-1])
                column1.append(int(row[0][:-1]))
                column2.append(int(row[1][:-1]))
    column1 = np.pad(column1, (0,max_length-len(column1)), 'constant')
    column2 = np.pad(column2, (0,max_length - len(column2)), 'constant')
    min=0 #3000
    max=10000
    column1, column2 = column1[min:max], column2[min:max]
    print(file_name)
    if is_mem:
        column = column2
    else:
        column = column1

    print("avg:", np.average(column), "max:", np.max(column), "median:", np.median(column))
    return column1, column2


fig = plt.figure(figsize=(8, 6))

# files = ["0-fp16", "0-fp32", "25-fp32", "50-fp32", "75-fp32"]
files = ["fp16-0", "fp32-0", "fp32-50", "fp32-75"]
# files = ["0-fp16", "0-fp32"]
for i, rate in enumerate(files):
    gpu, mem = read_columns(dir_path + "/utilization-" + str(rate) + ".csv")
    label=str(rate)+" %"
    if is_mem:
        plt.plot([x for x in range(len(mem))], mem, label=label, lw=3, marker=i)
    else:
        plt.plot([x for x in range(len(gpu))], gpu, label=label, lw=3, marker=i)

plt.xlabel('Time (msec)')

if is_mem:
    prefix = "Memory"
else:
    prefix = "GPU"

plt.ylabel(prefix + ' utilization (%)')
plt.grid()
plt.legend(loc='upper left', title="Compress:")

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
plt.show()
fig.savefig(prefix+"-utilization.pdf", bbox_inches='tight')
