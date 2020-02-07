import matplotlib

# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import csv
import os
from cnns.nnlib.utils.general_utils import get_log_time

print(matplotlib.get_backend())

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# configuration 1 figure
# # fontsize=20
# fontsize = 40
# legend_size = 25
# title_size = 40
# # width = 12
# # height = 5
# # lw = 3
# width = 8
# height = 8
# line_width = 4
# layout = "horizontal"  # "horizontal" or "vertical"


# configuration more figures
fontsize = 36
legend_size = 32
title_size = 40
width = 10
height = 7
line_width = 4
layout = "horizontal"  # "horizontal" or "vertical"

font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset + ".csv"
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=",", quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    try:
                        cols[column].append(float(row[column]))
                    except ValueError as ex:
                        print("Exception: ", ex)
    return cols


ylabel = "ylabel"
xlabel = "xlabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
column_nr = "column_nr"
labels = "labels"
legend_cols = "legend_cols"
xlim = "xlim"
ylim = "ylim"
is_log = "is_log"
legend_title = "legend_title"

carlini_cifar10 = {ylabel: "Accuracy (%)",
                   file_name: "distortionCarliniCifar3",
                   title: "C&W L$_2$ CIFAR-10",
                   legend_pos: "upper right",
                   # bbox: (0.0, 0.0),
                   column_nr: 12,
                   legend_cols: 2,
                   labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                   xlim: (0, 12),
                   ylim: (0, 100)}

carlini_imagenet = {ylabel: "Accuracy (%)",
                    file_name: "distortionCarliniImageNet",
                    title: "C&W L$_2$ ImageNet",
                    # legend_pos: "lower left",
                    legend_pos: "upper right",
                    # bbox: (0.0, 0.0),
                    column_nr: 12,
                    legend_cols: 2,
                    labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                    xlim: (0, 100),
                    ylim: (0, 100)}

carlini_imagenet_full = {ylabel: "Accuracy (%)",
                         file_name: "distortionCarliniImageNetFull4",
                         title: "C&W L$_2$ ImageNet",
                         # legend_pos: "lower left",
                         legend_pos: "upper right",
                         # bbox: (0.0, 0.0),
                         column_nr: 12,
                         legend_cols: 2,
                         labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace',
                                  'SVD'],
                         xlim: (0, 100),
                         ylim: (0, 100)}

pgd_cifar10 = {ylabel: "Accuracy (%)",
               file_name: "distortionPGDCifar",
               title: "PGD L$_{\infty}$ CIFAR-10",
               # legend_pos: "lower left",
               legend_pos: "upper right",
               # bbox: (0.0, 0.0),
               column_nr: 12,
               legend_cols: 2,
               labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
               xlim: (0, 12),
               ylim: (0, 100)}

random_pgd_cifar10 = {ylabel: "Accuracy (%)",
                      file_name: "distortionRandomPGDCifar",
                      # title: "PGD (random start) L$_{\infty}$ CIFAR-10",
                      title: "PGD L$_{\infty}$ CIFAR-10",
                      # legend_pos: "lower left",
                      legend_pos: "upper right",
                      # bbox: (0.0, 0.0),
                      column_nr: 12,
                      legend_cols: 2,
                      labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                      xlim: (0, 12),
                      ylim: (0, 100)}

random_pgd_cifar10_full = {ylabel: "Test accuracy (%)",
                           file_name: "distortionRandomPGDCifarFull",
                           # title: "PGD (random start) L$_{\infty}$ CIFAR-10",
                           title: "PGD L$_{\infty}$ CIFAR-10",
                           # legend_pos: "lower left",
                           legend_pos: "upper right",
                           # bbox: (0.0, 0.0),
                           column_nr: 12,
                           legend_cols: 2,
                           labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace',
                                    'SVD'],
                           xlim: (0, 12),
                           ylim: (0, 100)}

pgd_imagenet = {ylabel: "Accuracy (%)",
                file_name: "distortionPGDImageNet",
                title: "PGD L$_{\infty}$ ImageNet",
                # legend_pos: "lower left",
                legend_pos: "upper right",
                # bbox: (0.0, 0.0),
                column_nr: 12,
                legend_cols: 2,
                labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                xlim: (0, 100),
                ylim: (0, 100)}

fgsm_imagenet = {ylabel: "Accuracy (%)",
                 file_name: "distortionFGSMImageNet2",
                 title: "FGSM ImageNet",
                 # legend_pos: "lower left",
                 legend_pos: "upper right",
                 # bbox: (0.0, 0.0),
                 column_nr: 12,
                 legend_cols: 2,
                 labels: ['FC', 'CD', 'Unif', 'Gauss', 'Laplace', 'SVD'],
                 xlim: (0, 100),
                 ylim: (0, 100)}

robust_non_adaptive = {ylabel: "Test Accuracy (%)",
                       file_name: "distortion_robust_net_non_adaptive",
                       title: "C&W L$_2$ non-adaptive",
                       # legend_pos: "lower left",
                       legend_pos: "upper right",
                       # bbox: (0.0, 0.0),
                       column_nr: 6,
                       legend_cols: 1,
                       labels: ['plain', 'robust\n0.2 0.1', 'fft 50%'],
                       xlim: (0, 1.15),
                       ylim: (0, 100)}

robust_adaptive = {ylabel: "Test Accuracy (%)",
                   file_name: "distortion_robust_net2",
                   title: "C&W L$_2$ adaptive",
                   # legend_pos: "lower left",
                   legend_pos: "upper right",
                   # bbox: (0.0, 0.0),
                   column_nr: 6,
                   legend_cols: 1,
                   labels: ['plain', 'robust\n0.2 0.1', 'fft 50%'],
                   xlim: (0, 1.15),
                   ylim: (0, 100)}

robust_layers = {ylabel: "Test Accuracy (%)",
                 file_name: "distortion_robust_net_layers",
                 title: "C&W L$_2$ adaptive",
                 # legend_pos: "lower left",
                 legend_pos: "upper right",
                 # bbox: (0.0, 0.0),
                 column_nr: 8,
                 legend_cols: 1,
                 labels: ['plain', 'robust 0.2 0.1', 'robust 0.2 0.0',
                          'robust 0.3 0.0'],
                 xlim: (0, 1.15),
                 ylim: (0, 100)}

pni_robustnet = {ylabel: "Test Accuracy (%)",
                 file_name: "distortion_pni_robust_net",
                 title: "C&W L$_2$ adaptive",
                 # legend_pos: "lower left",
                 legend_pos: "upper right",
                 bbox: (-1.0, 0.0),
                 column_nr: 6,
                 legend_cols: 1,
                 labels: ['plain', 'PNI-W Adv\nResNet-20',
                          'RobustNet\nVGG16 2-1'],
                 xlim: (0, 1.15),
                 ylim: (0, 100)}

pni_robustnet2 = {ylabel: "Test Accuracy (%)",
                  file_name: "distortion_pni_robust_net2",
                  title: "C&W L$_2$ adaptive",
                  # legend_pos: "lower left",
                  legend_pos: "upper right",
                  bbox: (-1.0, 0.0),
                  column_nr: 8,
                  legend_cols: 2,
                  labels: ['plain', 'PNI-W Adv\nResNet-20',
                           'RobustNet\nVGG16 2-1',
                           'RobustNet\nResNet-20'],
                  xlim: (0, 1.15),
                  ylim: (0, 100)}

pni_robustnet3 = {ylabel: "Test Accuracy (%)",
                  file_name: "distortion_pni_robust_net3",
                  title: "C&W L$_2$ adaptive",
                  # legend_pos: "lower left",
                  legend_pos: "upper right",
                  bbox: (-1.0, 0.0),
                  column_nr: 10,
                  legend_cols: 2,
                  labels: ['plain', 'PNI-W Adv\nResNet-20',
                           'RobustNet\nVGG16 2-1',
                           'RobustNet\nResNet-20 2-1',
                           'RobustNet\nResNet-20 1-1'],
                  xlim: (0, 1.15),
                  ylim: (0, 100)}

pni_robustnet_adv_train = {ylabel: "Test Accuracy (%)",
                           file_name: "distortion_pni_robust_net5",
                           title: "C&W L$_2$ adaptive",
                           # legend_pos: "lower left",
                           legend_pos: "upper right",
                           bbox: (-1.0, 0.0),
                           column_nr: 10,
                           legend_cols: 1,
                           labels: ['plain',
                                    'RobustNet\nVGG16 2-1',
                                    'RobustNet\nResNet-20 2-1',
                                    'Adv. Train\nResNet-20',
                                    'PNI-W Adv\nResNet-20',
                                    ],
                           xlim: (0, 1.15),
                           ylim: (0, 100)}

pni_robustnet_adv_c_param = {ylabel: "Test Accuracy (%)",
                             xlabel: "C&W c parameter",
                             file_name: "distortion_pni_robust_net10",
                             title: "C&W L$_2$ adaptive",
                             # legend_pos: "lower left",
                             legend_pos: "upper right",
                             bbox: (-1.0, 0.0),
                             column_nr: 12,
                             legend_cols: 2,
                             labels: ['plain',
                                      'RobustNet\nVGG16 2-1',
                                      'RobustNet\nResNet-20 2-1',
                                      'Adv. Train\nResNet-20',
                                      'PNI-W Adv\nResNet-20',
                                      'RobustNet\nAdv. Train'
                                      ],
                             xlim: (0, 100),
                             ylim: (0, 100)}

pni_robustnet_adv_c_param2 = {ylabel: "Test Accuracy (%)",
                              xlabel: "C&W c parameter",
                              file_name: "distortion_pni_robust_net13",
                              title: "C&W L$_2$ adaptive",
                              legend_pos: "lower left",
                              # legend_pos: "upper right",
                              bbox: (-1.0, 0.0),
                              column_nr: 8,
                              legend_cols: 1,
                              labels: [
                                  'Adv. Train',
                                  'PNI-W Adv.',
                                  'RobustNet',
                                  'RobustNet Adv.'
                              ],
                              # labels: ['plain',
                              #          'RobustNet\nVGG16 2-1',
                              #          'RobustNet\nResNet-20 2-1',
                              #          'Adv. Train\nResNet-20',
                              #          'PNI-W Adv\nResNet-20',
                              #          'RobustNet\nAdv. Train'
                              #          ],
                              # xlim: (0, 100),
                              ylim: (0, 100),
                              is_log: True,
                              }

pni_robustnet_adv_train2 = {ylabel: "Test Accuracy (%)",
                            xlabel: "$L_2$ distortion",
                            file_name: "distortion_pni_robust_net7",
                            title: "C&W L$_2$ adaptive",
                            # legend_pos: "lower left",
                            legend_pos: "upper right",
                            bbox: (-1.0, 0.0),
                            column_nr: 12,
                            legend_cols: 2,
                            labels: ['plain',
                                     'RobustNet\nVGG16 2-1',
                                     'RobustNet\nResNet-20 2-1',
                                     'Adv. Train\nResNet-20',
                                     'PNI-W Adv\nResNet-20',
                                     'RobustNet\nAdv. Train'
                                     ],
                            xlim: (0, 1.6),
                            ylim: (0, 100)}

pni_robustnet_adv_train3 = {ylabel: "Test Accuracy (%)",
                            xlabel: "$L_2$ distortion",
                            file_name: "distortion_pni_robust_net12",
                            title: "C&W L$_2$ adaptive",
                            legend_pos: "lower left",
                            # legend_pos: "upper right",
                            bbox: (-1.0, 0.0),
                            column_nr: 8,
                            legend_cols: 1,
                            labels: [
                                'Adv. Train',
                                'PNI-W Adv.',
                                'RobustNet',
                                'RobustNet Adv.'
                            ],
                            # labels: ['plain',
                            #          'RobustNet\nVGG16 2-1',
                            #          'RobustNet\nResNet-20 2-1',
                            #          'Adv. Train\nResNet-20',
                            #          'PNI-W Adv\nResNet-20',
                            #          'RobustNet\nAdv. Train'
                            #          ],
                            ylim: (0, 100),
                            is_log: False}

pni_robustnet_adv_train_pgd = {ylabel: "Test Accuracy (%)",
                               xlabel: "$L_2$ distortion",
                               file_name: "distortion_pni_robust_net8",
                               title: "PGD L$_{\infty}$ adaptive",
                               # legend_pos: "lower left",
                               legend_pos: "upper right",
                               bbox: (-1.0, 0.0),
                               column_nr: 8,
                               legend_cols: 2,
                               labels: [
                                   'Adv. Train\nResNet-20',
                                   'PNI-W Adv\nResNet-20',
                                   'RobustNet\nResNet-20 2-1',
                                   'RobustNet\nAdv. Train'
                               ],
                               xlim: (0.9, 1.7),
                               ylim: (0, 100)}

pni_robustnet_adv_train_pgd_iters = {ylabel: "Test Accuracy (%)",
                                     xlabel: '# of PGD iterations',
                                     file_name: "distortion_pni_robust_net9",
                                     title: "PGD L$_{\infty}$ adaptive",
                                     # legend_pos: "lower left",
                                     legend_pos: "upper right",
                                     bbox: (-1.0, 0.0),
                                     column_nr: 8,
                                     legend_cols: 2,
                                     labels: [
                                         'Adv. Train\nResNet-20',
                                         'PNI-W Adv\nResNet-20',
                                         'RobustNet\nResNet-20 2-1',
                                         'RobustNet\nAdv. Train'
                                     ],
                                     xlim: (0, 1000),
                                     ylim: (0, 100)}

pni_robustnet_adv_train_pgd_dist_linf = {ylabel: "Test Accuracy (%)",
                                         xlabel: '$L_\infty$ distortion x $10^{-6}$',
                                         file_name: "distortion_pni_robust_net11",
                                         title: "PGD L$_{\infty}$ adaptive",
                                         # legend_pos: "lower left",
                                         legend_pos: "upper right",
                                         bbox: (-1.0, 0.0),
                                         column_nr: 8,
                                         legend_cols: 2,
                                         labels: [
                                             'Adv. Train\nResNet-20',
                                             'PNI-W Adv\nResNet-20',
                                             'RobustNet\nResNet-20 2-1',
                                             'RobustNet\nAdv. Train'
                                         ],
                                         xlim: (0, 1000),
                                         ylim: (0, 100)}

pni_robustnet_adv_train_pgd_iters2 = {ylabel: "Test Accuracy (%)",
                                      xlabel: '# of PGD iterations',
                                      file_name: "distortion_pni_robust_net9",
                                      title: "PGD L$_{\infty}$ adaptive",
                                      # legend_pos: "lower left",
                                      legend_pos: "upper right",
                                      bbox: (-1.0, 0.0),
                                      column_nr: 8,
                                      legend_cols: 1,
                                      labels: [
                                          'Adv. Train',
                                          'PNI-W Adv.',
                                          'RobustNet',
                                          'RobustNet Adv.'
                                      ],
                                      # labels: [
                                      #     'Adv. Train\nResNet-20',
                                      #     'PNI-W Adv\nResNet-20',
                                      #     'RobustNet\nResNet-20 2-1',
                                      #     'RobustNet\nAdv. Train'
                                      # ],
                                      ylim: (0, 100),
                                      is_log: True}

pni_robustnet_adv_train_pgd_dist_linf2 = {ylabel: "Test Accuracy (%)",
                                          xlabel: '$L_\infty$ distortion x $10^{-6}$',
                                          file_name: "distortion_pni_robust_net11",
                                          title: "PGD L$_{\infty}$ adaptive",
                                          legend_pos: "lower left",
                                          # legend_pos: "upper right",
                                          bbox: (-1.0, 0.0),
                                          column_nr: 8,
                                          legend_cols: 1,
                                          labels: [
                                              'Adv. Train',
                                              'PNI-W Adv.',
                                              'RobustNet',
                                              'RobustNet Adv.'
                                          ],
                                          ylim: (0, 100),
                                          is_log: False,
                                          }

robust_non_adaptive2 = {ylabel: "Test Accuracy (%)",
                        file_name: "distortion_robust_net_non_adaptive",
                        title: "C&W L$_2$ non-adaptive",
                        # legend_pos: "lower left",
                        legend_pos: "upper right",
                        # bbox: (0.0, 0.0),
                        column_nr: 6,
                        legend_cols: 1,
                        labels: ['PlainNet', 'RobustNet', 'FC'],
                        xlim: (-0.05, 1.15),
                        ylim: (0, 100),
                        xlabel: '$L_2$ distortion',
                        is_log: False}

robust_adaptive2 = {ylabel: "Test Accuracy (%)",
                    file_name: "distortion_robust_net2",
                    title: "C&W L$_2$ adaptive",
                    # legend_pos: "lower left",
                    legend_pos: "upper right",
                    # bbox: (0.0, 0.0),
                    column_nr: 6,
                    legend_cols: 1,
                    # labels: ['plain', 'robust\n0.2 0.1', 'fft 50%'],
                    labels: ['PlainNet', 'RobustNet', 'FC'],
                    xlim: (-0.05, 1.15),
                    ylim: (0, 100),
                    xlabel: '$L_2$ distortion',
                    is_log: False}

train_vs_inference = {
    ylabel: "Test Accuracy (%)",
    file_name: "train_vs_test_perturbation2",
    # title: "C&W L$_2$ adaptive",
    title: "ParamNet",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 8,
    legend_cols: 1,
    labels: ['test 0.01', 'train 0.01',
             'test 0.02', 'train 0.02'],
    # xlim: (0, 1.15),
    ylim: (0, 100),
    xlabel: '$L_2$ distortion',
    is_log: False,
    # legend_title: 'ParamNet:',
}

train_vs_inference3 = {
    ylabel: "Test Accuracy (%)",
    file_name: "train_vs_test_perturbation3",
    # title: "C&W L$_2$ adaptive",
    title: "ParamNet",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 12,
    legend_cols: 1,
    labels: ['test 0.01', 'train 0.01',
             'test 0.02', 'train 0.02',
             'test 0.07', 'train 0.07'],
    # xlim: (0, 1.15),
    ylim: (0, 100),
    xlabel: '$L_2$ distortion',
    is_log: False,
    # legend_title: 'ParamNet:',
}

robust_layers_dp = {
    ylabel: "Test Accuracy (%)",
    file_name: "distortion_robust_net_layers",
    # title: "C&W L$_2$ adaptive",
    title: "RobustNet",
    # legend_pos: "lower left",
    legend_pos: "upper right",
    # bbox: (0.0, 0.0),
    column_nr: 8,
    legend_cols: 1,
    labels: ['0.0 0.0', '0.2 0.1', '0.2 0.0',
             '0.3 0.0'],
    xlim: (-0.05, 1.15),
    ylim: (0, 100),
    xlabel: '$L_2$ distortion',
    is_log: False,
    # legend_title: 'RobustNet:',
}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["o", "v", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]

# datasets = [
#     random_pgd_cifar10_full,
#     carlini_cifar10,
#     carlini_imagenet_full,
# ]
# datasets = [carlini_cifar10,
#             carlini_imagenet,
#             # carlini_imagenet_full,
#             # pgd_cifar10,
#             random_pgd_cifar10,
#             pgd_imagenet,
#             fgsm_imagenet,
#             ]

# datasets = [
#     random_pgd_cifar10_full,
#     carlini_imagenet_full,
# ]

# datasets = [robust_layers]
# datasets = [pni_robustnet_adv_train2]
# datasets = [pni_robustnet_adv_train_pgd]

# datasets = [
#     pni_robustnet_adv_c_param,
#     pni_robustnet_adv_train2,
#     pni_robustnet_adv_train_pgd_iters,
#     pni_robustnet_adv_train_pgd,
# ]

# datasets = [
#     pni_robustnet_adv_c_param,
#     pni_robustnet_adv_train2,
#     pni_robustnet_adv_train_pgd_iters,
#     pni_robustnet_adv_train_pgd_dist_linf,
# ]

# first distortion
# datasets = [
#     random_pgd_cifar10_full,
#     carlini_imagenet_full,
# ]

# non adaptive vs adaptive
datasets = [robust_non_adaptive2, robust_adaptive2]

# train vs test + where to place noise layer
# datasets = [train_vs_inference3, robust_layers_dp]

# last figure
# datasets = [
#     pni_robustnet_adv_c_param2,
#     pni_robustnet_adv_train3,
#     pni_robustnet_adv_train_pgd_iters2,
#     pni_robustnet_adv_train_pgd_dist_linf2,
# ]

fig = plt.figure(figsize=(len(datasets) * width, height))

for j, dataset in enumerate(datasets):
    if layout == "vertical":
        ax = plt.subplot(len(datasets), 1, j + 1)
    else:
        ax = plt.subplot(1, len(datasets), j + 1)
    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)

    print("col 0: ", cols[0])
    print("col 1: ", cols[1])

    for col in range(0, columns, 2):
        i = col // 2
        print('i: ', i)
        plt.plot(cols[col], cols[col + 1],
                 label=f"{dataset[labels][i]}",
                 lw=line_width,
                 color=colors[i],
                 linestyle=linestyles[i],
                 marker=markers[i],
                 ms=8)

    plt.grid()
    if legend_title in dataset:
        legend_title_str = dataset[legend_title]
    else:
        legend_title_str = None
    plt.legend(loc=dataset[legend_pos],
               ncol=dataset[legend_cols],
               frameon=False,
               prop={'size': legend_size},
               columnspacing=1.0,
               # bbox_to_anchor=dataset[bbox],
               title=legend_title_str,
               )
    if xlabel in dataset:
        xlabel_str = dataset[xlabel]
    else:
        xlabel_str = '$L_2$ distortion'
    plt.xlabel(xlabel_str)
    plt.title(dataset[title], fontsize=title_size)
    if j == 0:
        plt.ylabel(dataset[ylabel])
    else:
        pass
        # ax.axes.get_yaxis().set_visible(False)
        # ax.set_yticklabels([])
    plt.ylim(dataset[ylim])
    if xlim in dataset:
        plt.xlim(dataset[xlim])
    if is_log in dataset and dataset[is_log]:
        plt.xscale('log', basex=10)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.3)
format = "pdf"  # "pdf" or "png"
destination = dir_path + "/" + "distortion_pni_robustnet_" + get_log_time() + "." + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            transparent=False
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
