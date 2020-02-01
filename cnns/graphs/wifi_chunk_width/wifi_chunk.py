import matplotlib
import matplotlib.pyplot as plt
import os

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)

legend_size = 20
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
static_x = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
classes_2 = (
    85.01, 87.15, 91.19, 94.92, 97.01, 98.09, 98.92, 99.15, 99.21, 99.28, 99.42,
    99.60)
classes_3 = (32.46,
             53.71646407,
             65.50106259,
             63.44205055,
             72.93496023,
             74.83560951,
             80.26110958,
             87.90859934,
             96.59671708,
             99.63,
             99.95,
             99.98705502,
             )

static_x_4 = (
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    2,
    1,
)
classes_4 = (
    99.01719902,
    97.22703362,
    91.17772109,
    80.57609183,
    70.90793141,
    62.22395101,
    56.65497521,
    51.64945676,
    42.78901156,
    25,
)

classes_4 = [x for x in reversed(classes_4)]
static_x_4 = [x for x in reversed(static_x_4)]

static_5 = (
    1024,
    512,
    128,
    32,
    8,
    2,
    1,
)

classes_5 = (
    98.83349374,
    96.85334873,
    78.68956559,
    57.04521948,
    43.29458844,
    34.34693261,
    20,
)

classes_5 = [x for x in reversed(classes_5)]
static_5 = [x for x in reversed(static_5)]

plt.plot(static_x, classes_2, label='2 classes', lw=3, marker='o',
         color=get_color(MY_BLUE))
plt.plot(static_x, classes_3, label='3 classes', lw=3, marker='v',
         color=get_color(MY_RED))
plt.plot(static_x_4, classes_4,
         label='4 classes', lw=3, marker='+',
         color=get_color(MY_GREEN))
plt.plot(static_5, classes_5,
         label='5 classes', lw=3, marker='+',
         color=get_color(MY_BLACK))
# plt.plot(mix_x, mix_y, label='energy first + static rest', lw=2, marker='v',
#          color=get_color(MY_ORANGE))
# plt.plot(energy_x, energy_y, label='energy based compression', lw=2, marker='s',
#          color=get_color(MY_GREEN))

plt.xlabel('Chunk size')
plt.ylabel('Test accuracy (%)')
plt.xscale('log', basex=2)
plt.xlim(1, 2048)
plt.ylim(30, 100)
plt.grid()
plt.legend(loc=legend_position,
           frameon=frameon,
           prop={'size': legend_size},
           # bbox_to_anchor=bbox_to_anchor
           )

plt.show()
fig.savefig(dir_path + "/" + "test-accuracy-chunk-size5.pdf",
            bbox_inches='tight')
plt.close()
