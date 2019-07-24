import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
import numpy as np

print(matplotlib.get_backend())

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# fontsize=20
size=18
fontsize = size
legend_size = size
label_size = 10

font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

labels = ['ED:LOS', 'ED:NLOS', 'AC:LOS', 'AC:NLOS', 'ML:LOS', 'ML:NLOS']
v6F = [98.0, 84.0, 99.0, 95.0, 99.70, 99.12]
v10F = [96.0, 76.0, 98.3, 94.0, 99.84, 99.57]
v15F = [84.0, 71.0, 98.0, 91.0, 99.98, 97.76]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig_width = 35 * width
fig_height = 20 * width
lw = 4

# plt.figure(figsize=(width, height))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
rects1 = ax.bar(x - width, v6F, width, label='6F', color=get_color(MY_BLUE))
rects2 = ax.bar(x, v10F, width, label='10F', color=get_color(MY_RED), hatch='/')
rects3 = ax.bar(x + width, v15F, width, label='15F',
                color=get_color(MY_GREEN), hatch='*')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Successful Detection (%)')
ax.set_ylim(0, 118)
ax.set_xlabel('Types of CSAT approach')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=label_size)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.legend(loc="upper center", ncol=3, frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )

fig.tight_layout()
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "wifi-n-los-one-model." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            # transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
# plt.show()
plt.close()
