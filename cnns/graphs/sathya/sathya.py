import matplotlib
import matplotlib.pyplot as plt
import os

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)

legend_size = 16
legend_position = 'best'
frameon = False
bbox_to_anchor = (0, -0.1)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


font = {'size': 20}
title_size = 16
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

fig = plt.figure(figsize=(8, 6))

# plt.title("Model accuracy for a given chunk size", fontsize=title_size)
static_x = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
static_y = (
85.01, 87.15, 91.19, 94.92, 97.01, 98.09, 98.92, 99.15, 99.21, 99.28, 99.42,
99.60)

plt.plot(static_x, static_y, label='static compression', lw=2, marker='o',
         color=get_color(MY_BLUE))
# plt.plot(mix_x, mix_y, label='energy first + static rest', lw=2, marker='v',
#          color=get_color(MY_ORANGE))
# plt.plot(energy_x, energy_y, label='energy based compression', lw=2, marker='s',
#          color=get_color(MY_GREEN))

plt.xlabel('Chunk size')
plt.ylabel('Test accuracy (%)')
plt.grid()
# plt.legend(loc=legend_position, frameon=frameon, prop={'size': legend_size},
#            bbox_to_anchor=bbox_to_anchor)

plt.show()
fig.savefig(dir_path + "/" + "test-accuracy-chunk-size.pdf",
            bbox_inches='tight')
