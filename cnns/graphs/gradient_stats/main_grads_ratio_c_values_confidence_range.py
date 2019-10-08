import matplotlib
import numpy as np

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
MY_BLUE = (57, 106, 177)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_RED = (204, 37, 41)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)
MY_VIOLET = (107, 76, 154)
MY_BROWN = (146, 36, 40)
MY_OWN = (25, 150, 10)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


# fontsize=20
fontsize = 36
legend_size = fontsize
title_size = fontsize
font = {'size': fontsize}
matplotlib.rc('font', **font)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir path: ", dir_path)

GPU_MEM_SIZE = 16280


def read_columns(dataset, columns=5):
    file_name = dir_path + "/" + dataset
    with open(file_name) as csvfile:
        data = csv.reader(csvfile, delimiter=";", quotechar='|')
        cols = []
        for column in range(columns):
            cols.append([])

        for i, row in enumerate(data):
            if i > 0:  # skip header
                for column in range(columns):
                    try:
                        # print('column: ', column)
                        cols[column].append(float(row[column]))
                    except ValueError as ex:
                        pass
                        cols[column].append(row[column])
                        # print("Exception: ", ex)
    return cols


ylabel = "ylabel"
title = "title"
legend_pos = "center_pos"
bbox = "bbox"
file_name = "file_name"
column_nr = "column_nr"
labels = "labels"
legend_cols = "legend_cols"
xlim = "xlim"
ylim = "ylim"

recovered_0_001 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-01-43-45-115755_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.001, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_001 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.001, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_001_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-21-42-20-803744_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.001, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_001_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-21-42-20-803744_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.001, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-01-44-43-896068_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-01-44-43-896068_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01_conf_1 = {  # ylabel: "L2 adv",
    file_name: "2019-09-25-20-18-04-052680_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=1'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01_conf_1 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-44-43-896068_grad_stats_False-cifar10.csv",
    file_name: "2019-09-25-20-18-04-052680_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=1'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01_conf_10 = {
    file_name: "2019-09-25-20-18-48-322123_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=10'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01_conf_10 = {
    file_name: "2019-09-25-20-18-48-322123_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=10'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01_conf_100 = {
    file_name: "2019-09-25-20-19-39-790685_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=100'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01_conf_100 = {
    file_name: "2019-09-25-20-19-39-790685_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=100'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01_conf_200 = {
    file_name: "2019-09-25-20-21-35-669154_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=200'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01_conf_200 = {
    file_name: "2019-09-25-20-21-35-669154_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=200'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_01_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-24-01-44-43-896068_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-21-47-29-622907_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.01, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_01_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-44-43-896068_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-21-47-29-622907_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.01, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_1 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-01-45-37-590896_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.1, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_1 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-01-45-37-590896_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.1, confidece=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_0_1_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-24-01-45-37-590896_grad_stats_True-cifar10.csv",
    file_name: "2019-09-24-21-47-48-129153_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=0.1, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_0_1_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-45-37-590896_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-21-47-48-129153_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=0.1, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_1_0 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_True-cifar10.csv",
    file_name: '2019-09-24-01-46-44-250879_grad_stats_True-cifar10.csv',
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=1.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_1_0 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-01-46-44-250879_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=1.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_1_0_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-46-44-250879_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-21-48-09-781282_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=1.0, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_1_0_conf_1000 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-46-44-250879_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-21-48-09-781282_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=1.0, confidence=1000'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_10_0 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_True.csv",
    # file_name: "recovered.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_True-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_True-cifar10.csv",
    # file_name: '2019-09-24-01-46-44-250879_grad_stats_True-cifar10.csv',
    file_name: '2019-09-24-18-28-05-302979_grad_stats_True-cifar10.csv',
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=10.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_10_0 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-18-28-05-302979_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=10.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_100_0 = {  # ylabel: "L2 adv",
    file_name: "2019-09-24-18-29-16-691092_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=100.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_100_0 = {  # ylabel: "L2 adv",
    # file_name: "2019-09-09-18-28-04-319343_grad_stats_False.csv",
    # file_name: "not_recovered.csv",
    # file_name: "2019-09-11-22-09-14-647707_grad_stats_False.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-19-21-51-01-222075_grad_stats_True.csv",
    # file_name: "2019-09-21-00-04-48-125028_grad_stats_False-cifar10.csv",
    # file_name: "2019-09-24-01-43-45-115755_grad_stats_False-cifar10.csv",
    file_name: "2019-09-24-18-29-16-691092_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=100.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_1000_0 = {  # ylabel: "L2 adv",
    file_name: "2019-09-24-18-29-54-114768_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=1000.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_1000_0 = {  # ylabel: "L2 adv",
    file_name: "2019-09-24-18-29-54-114768_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=1000.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

recovered_10000_0 = {  # ylabel: "L2 adv",
    file_name: "2019-09-24-18-30-30-626095_grad_stats_True-cifar10.csv",
    title: "recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['recovered c=10000.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

not_recovered_10000_0 = {  # ylabel: "L2 adv",
    file_name: "2019-09-24-18-30-30-626095_grad_stats_False-cifar10.csv",
    title: "not recovered",
    # legend_pos: "lower left",
    legend_pos: "lower right",
    # bbox: (0.0, 0.0),
    column_nr: 62,
    legend_cols: 1,
    labels: ['not recovered c=10000.0, confidence=0'],
    xlim: (0, 100),
    ylim: (0, 100)}

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN]]
markers = ["+", "o", "v", "s", "D", "^", "+", 'o', 'v', '+']
linestyles = [":", "-", "--", ":", "-", "--", "-", "--", ':', ':']

datasets = [
    # recovered_0_001,
    # not_recovered_0_001,
    # recovered_0_001_conf_1000,
    # not_recovered_0_001_conf_1000,
    recovered_0_01,
    not_recovered_0_01,
    recovered_0_01_conf_1000,
    not_recovered_0_01_conf_1000,
    recovered_0_01,
    not_recovered_0_01,
    # recovered_0_01_conf_200,
    # not_recovered_0_01_conf_200,
    # recovered_0_01_conf_100,
    # not_recovered_0_01_conf_100,
    # recovered_0_01_conf_10,
    # not_recovered_0_01_conf_10,
    # recovered_0_01_conf_1,
    # not_recovered_0_01_conf_1,
    # recovered_0_01,
    # not_recovered_0_01,
    # recovered_0_01_conf_1,
    # not_recovered_0_01_conf_1,
    # recovered_0_1,
    # not_recovered_0_1,
    # recovered_0_1_conf_1000,
    # not_recovered_0_1_conf_1000,
    # recovered_1_0,
    # not_recovered_1_0,
    # recovered_1_0_conf_1000,
    # not_recovered_1_0_conf_1000,
    # recovered_10_0,
    # not_recovered_10_0,
    # recovered_100_0,
    # not_recovered_100_0,
    # recovered_1000_0,
    # not_recovered_1000_0,
    # recovered_10000_0,
    # not_recovered_10000_0,
    # yes_1070,
    # no_1070,
    # carlini_imagenet,
    # pgd_cifar10,
    # random_pgd_cifar10,
    # pgd_imagenet,
    # fgsm_imagenet,
]

# width = 12
# height = 5
# lw = 3
decimals = 3
fig_size = 10
width = 15
height = 15
line_width = 4
layout = "horizontal"  # "horizontal" or "vertical"
# limit = 4096
nrows = 1
assert len(datasets) % 2 == 0
ncols = len(datasets) // 2 // nrows

fig = plt.figure(figsize=(ncols * width, nrows * height))
dist_recovered = None

for j, dataset in enumerate(datasets):

    print("dataset: ", dataset)
    columns = dataset[column_nr]
    cols = read_columns(dataset[file_name], columns=columns)
    if j < 2:
        is_org_image = True
    else:
        is_org_image = False

    if is_org_image:
        org_col = 42
        adv_col = 40
        print(f'cols[{org_col}][0]: ', cols[org_col][0])
        assert cols[org_col][0] == 'l2_norm_original_correct'
        print(f'cols[{adv_col}][0]: ', cols[adv_col][0])
        assert cols[adv_col][0] == 'l2_norm_original_adv'
        org_grad = cols[org_col + 1]
        adv_grad = cols[adv_col + 1]
        x_grad = org_grad
        y_grad = adv_grad
    else:
        org_col = 30
        adv_col = 28
        print(f'cols[{org_col}][0]: ', cols[org_col][0])
        assert cols[org_col][0] == 'l2_norm_adv_correct'
        print(f'cols[{adv_col}][0]: ', cols[adv_col][0])
        assert cols[adv_col][0] == 'l2_norm_adv_adv'
        org_grad = cols[org_col + 1]
        adv_grad = cols[adv_col + 1]
        x_grad = adv_grad
        y_grad = org_grad

    l2_dist_col = 60
    print(f'cols[{l2_dist_col}][0]: ', cols[l2_dist_col][0])
    assert cols[l2_dist_col][0] == 'z_l2_dist_adv_org_image'

    # print("col org: ", org_grad)
    # print("col adv: ", adv_grad)
    # print('col length: ', len(adv_grad))

    l2_dist = cols[l2_dist_col + 1]
    # print('col l2_dist: ', l2_dist)
    avg_l2_dist = np.average(l2_dist)
    print('recovered count: ', len(adv_grad))
    print('avg l2 dist: ', avg_l2_dist)
    print('avg org grad: ', np.average(org_grad))
    print('avg adv grad: ', np.average(adv_grad))

    print(len(adv_grad))
    print(avg_l2_dist)
    print(np.average(org_grad))
    print(np.average(adv_grad))

    if j % 2 == 0:
        plt.subplot(nrows, ncols, j // 2 + 1)
        dist_recovered = str(np.around(avg_l2_dist, decimals=decimals))
    label = dataset[labels][0].replace(', ', '\n')
    plt.plot(x_grad, y_grad,
             label=label,
             # lw=line_width,
             color=colors[j % len(colors)],
             # linestyle=linestyles[j],
             linestyle='None',
             marker=markers[j % len(markers)])

    if j % 2 == 1:
        dist_not_recovered = str(np.around(avg_l2_dist, decimals=decimals))
        plt.grid()
        plt.legend(loc=dataset[legend_pos], ncol=dataset[legend_cols],
                   frameon=False,
                   prop={'size': legend_size},
                   # bbox_to_anchor=dataset[bbox]
                   )
        if is_org_image:
            plt.ylabel('$L_2$ of gradient for org image and adv class')
            plt.xlabel('$L_2$ of gradient for org image and org class')
            plt.title('The original image')
        else:
            plt.ylabel('$L_2$ of gradient for adv image and org class')
            plt.xlabel('$L_2$ of gradient for adv image and adv class')
            plt.title(
                f'$L_2$ distance\nrecovered: {dist_recovered}, not recovered: {dist_not_recovered}',
                fontsize=title_size)

        plt.xlim((0, 50))
        plt.ylim((0, 50))
        # plt.xscale('log', basex=10)
        # plt.yscale('log', basey=10)

# plt.gcf().autofmt_xdate()
# plt.xticks(rotation=0)
# plt.interactive(False)
# plt.imshow()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
format = "png"  # "pdf" or "png"
destination = dir_path + "/" + "main_grads_ratio_c_values_confidence_range_" + get_log_time() + '.' + format
print("destination: ", destination)
fig.savefig(destination,
            bbox_inches='tight',
            transparent=True
            )
# plt.show(block=False)
# plt.interactive(False)
plt.close()
