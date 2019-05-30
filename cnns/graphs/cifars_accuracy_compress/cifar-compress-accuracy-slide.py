import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)

legend_size = 16
legend_position = 'bottom left'
frameon = False
bbox_to_anchor = (0, -0.1)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


font = {'size': 20}
title_size = 16
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))

fig = plt.figure(figsize=(8, 6))

plt.subplot(1, 1, 1)

resnet_x = (
    0, 8.3855401, 11.87663269, 14.5, 17.8, 21.0941708, 24, 26, 29.26308507,
    39.203635, 49, 53.53477122, 67.50730502, 77.77388775)
resnet_y = (
    93.69, 93.42, 93.12, 93, 93.06, 93.24, 93.16, 93.2, 92.89, 92.61, 91.95, 91.64, 89.71, 87.97)

densenet_x = (
    0, 9.732438751, 11.74155899, 16.71815462, 20.25832864, 23.932799,
    29.48016456, 40.12626288, 48.87082897, 50.91781047, 77.82953128)
densenet_y = (
    75.3, 75.28, 74.99, 74.63, 74.25, 74.2, 73.66, 72.26, 71.53, 71.18, 63.95)

plt.plot(resnet_x, resnet_y, label='ResNet-18 on CIFAR-10', lw=3, marker='o',
         color=get_color(MY_RED))
# plt.plot(mix_x, mix_y, label='energy first + static rest', lw=2, marker='v',
#          color=get_color(MY_ORANGE))
plt.plot(densenet_x, densenet_y, label='DenseNet-121 on CIFAR-100', lw=3,
         marker='s', color=get_color(MY_GREEN))

plt.xlabel('Compression ratio (%)')
plt.ylabel('Test accuracy (%)')
plt.grid()
plt.legend(loc=legend_position, frameon=frameon, prop={'size': legend_size},
           bbox_to_anchor=bbox_to_anchor)

plt.show()
fig.savefig(dir_path + "/" + "cifars-accuracy-compress-slide.pdf",
            bbox_inches='tight')
