import numpy as np
import os
import pandas as pd
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

ylabel_size = 25
font = {'size': 30}
matplotlib.rc('font', **font)
lw = 4
fontsize = 25
legend_size = fontsize
title_size = fontsize
font = {'size': fontsize}
matplotlib.rc('font', **font)

legend_position = 'right'
frameon = False
bbox_to_anchor = (0.0, -0.1)

delimiter = ','

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


line_width = 4
colors = [get_color(color) for color in
          [MY_RED, MY_BLUE, MY_RED, MY_GREEN, MY_BLACK, MY_GOLD,
           MY_VIOLET, MY_OWN, MY_BROWN, MY_GREEN]]
markers = ["o", "+", "^", "v", "D", "^", "+", 'o', 'v', '+']
linestyles = ["-", "--", ":", "--", "-", "--", "-", "--", ':', ':']


def check_image_indexes(data_all, step):
    image_indexes = get_column(data_all, col_nr=13, col_name='image_index',
                               dtype=np.int)
    i = 1
    j = 0
    for val in image_indexes:
        if i != val:
            print(f"{i} has to be equal {val}")
            i += 1
        if i == step:
            print('j: ', j)
            i = 0
        i += 1


stats = {
    'avg': np.nanmean,
    # 'median': np.median,
    # 'std': np.std,
    # 'min': np.min,
    # 'max': np.max,
}


def get_stats(vals):
    results = {}
    for key, op in stats.items():
        results[key] = op(vals)
    return results


def get_column(data_all, col_nr, col_name, dtype=np.float):
    # col_name_from_col_nr = data_all[col_nr - 1][0]
    # print('col name: ', col_name_from_col_nr)
    assert col_name == col_name
    vals = np.asarray(
        data_all.iloc[:, col_nr],
        dtype=dtype)
    return vals


def get_col_val(data_all, col_name, dtype=np.float):
    W = data_all.shape[1]
    col_nr = None
    for i in range(W):
        data_col_name = data_all.iloc[0, i]
        # print('data_col_name: ', data_col_name)
        if data_col_name == col_name:
            col_nr = i + 1
    if col_nr is None:
        raise Exception(f'Column with name: {col_name} not found.')
    return get_column(data_all=data_all, col_nr=col_nr,
                      col_name=col_name, dtype=dtype)


def get_col_vals(data_all, col_names, dtype=np.float):
    col_vals = []
    for col_name in col_names:
        col_val = get_col_val(data_all=data_all, col_name=col_name, dtype=dtype)
        col_vals.append(col_val)
    return col_vals


def plot_graph(recovered_org_grad, recovered_2nd_grad,
               not_recovered_org_grad, not_recovered_2nd_grad):
    i = 0
    plt.scatter(recovered_org_grad, recovered_2nd_grad,
                label='recovered',
                lw=line_width,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                # linestyle='None',
                marker=markers[i % len(markers)])
    i += 1
    plt.scatter(not_recovered_org_grad, not_recovered_2nd_grad,
                label='not recovered',
                lw=line_width,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                # linestyle='None',
                marker=markers[i % len(markers)])

    # plt.title('Scatter plot pythonspot.com')
    plt.xlabel('org grad')
    plt.ylabel('2nd highest org grad')
    plt.legend()
    plt.show()


def plot_results(results):
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(2, 1, 1)
    plt.title("Carlini-Wagner $L_2$ attack", fontsize=title_size)

    x = results[:, 0]
    y_1 = results[:, 1] * 100

    plt.plot(x, y_1,
             # label='train on SVD representation (3 channels, V*D*U is p by p)',
             label=f"% lowest gradient = highest confidence",
             lw=lw,
             linestyle=":",
             marker='',
             color=get_color(MY_BLUE)
             )


    # condition number
    y_2 = results[:, 2] * 100

    plt.plot(x, y_2,
             # label='train on SVD representation (3 channels, V*D*U is p by p)',
             label="% adversarial examples found",
             lw=lw,
             linestyle='-',
             marker='o',
             color=get_color(MY_RED)
             )

    # plt.xlabel("C&W $L_2$ attack strength")
    plt.ylabel("Normalized %", fontsize=ylabel_size)
    plt.xscale('log', basex=10)
    plt.ylim(0, 100)
    # plt.xticks(np.arange(min(x), max(x) + 5000, 5000))
    # plt.xlim(0, 23000)
    # plt.grid()
    plt.legend(loc='lower right',
               frameon=frameon,
               prop={'size': legend_size},
               # bbox_to_anchor=bbox_to_anchor,
               ncol=1,
               )

    # plot the adversarial distance
    plt.subplot(2, 1, 2)
    y_3 = results[:, -1]
    plt.plot(x, y_3,
             # label='train on SVD representation (3 channels, V*D*U is p by p)',
             label="Adversarial $L_2$ distance",
             lw=lw,
             linestyle='-',
             marker='^',
             color=get_color(MY_BLACK)
             )

    plt.xlabel("C&W $L_2$ attack strength")
    plt.ylabel("$L_2$ distance", fontsize=ylabel_size)
    plt.xscale('log', basex=10)
    # plt.ylim(0, 100)
    # plt.xticks(np.arange(min(x), max(x) + 5000, 5000))
    # plt.xlim(0, 23000)
    # plt.grid()
    plt.legend(loc='lower right',
               frameon=frameon,
               prop={'size': legend_size},
               # bbox_to_anchor=bbox_to_anchor,
               ncol=1,
               )

    fig.tight_layout()
    # plt.show()
    format = "pdf"  # "png" or "pdf"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = dir_path + "/" + "carlini_wagner_attack_strength_confidence_norm_gradient_"
    file_name += "_" + get_log_time() + "." + format
    fig.savefig(file_name,
                bbox_inches='tight',
                transparent=False)
    plt.close()


def get_file_stats(file_name, adv_strength, results, class_type='adv'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, file_name)
    start_from_line = None
    data_all = pd.read_csv(data_path, header=start_from_line, sep=delimiter)
    # print('shape of data all: ', data_all.shape)
    # print(data_all.head(5))
    H, W = data_all.shape
    correctly_classified = H
    # print('correctly_classified: ', correctly_classified)

    data_all = data_all[data_all.iloc[:, 4] == 'adv_class']

    params = [0.001, 0.002, 0.004, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
              0.08, 0.1, 0.5, 1.0]

    step = 939
    # data_all = data_all[7 * step:8 * step]
    H, W = data_all.shape

    col_names = []
    for class_nr in range(10):
        col_names += ['l2_norm_' + class_type + '_class_' + str(class_nr)]
    norm_vals = get_col_vals(data_all=data_all, col_names=col_names)

    target_classes = get_col_val(data_all=data_all,
                                 col_name=class_type + '_class',
                                 dtype=np.int)

    recovered = get_col_val(data_all=data_all,
                            col_name='is_gauss_recovered',
                            dtype=np.bool)
    # print('total recovered: ', np.nansum(recovered))

    dist_adv_org = get_col_val(data_all=data_all,
                               col_name='z_l2_dist_adv_org_image',
                               dtype=np.float)
    # print('averge l2 dist adv org: ', np.nanmean(dist_adv_org))

    class_type = 'org' if class_type == 'original' else 'adv'
    col_names = []
    for class_nr in range(10):
        col_names += [class_type + '_confidence_class_' + str(class_nr)]
    conf_vals = get_col_vals(data_all=data_all, col_names=col_names)

    recovered_org_grad = []
    recovered_2nd_grad = []

    not_recovered_org_grad = []
    not_recovered_2nd_grad = []

    count_lowest = 0
    count_highest = 0
    for row_nr in range(H):
        class_nr = target_classes[row_nr]
        confs = []
        for conf_val in conf_vals:
            confs.append(conf_val[row_nr])
        max_conf = np.argmax(confs)
        assert max_conf == class_nr
        norms = []
        for norm_val in norm_vals:
            norms.append(norm_val[row_nr])
        min_class = np.argmin(norms)
        if min_class == class_nr:
            count_lowest += 1
        max_class = np.argmax(norms)
        if max_class == class_nr:
            count_highest += 1

        # norms = (norms - np.average(norms)) / np.std(norms)

        norm_org = norms[class_nr]
        norms[class_nr] = np.max(norms)
        norm_2nd = np.min(norms)

        if recovered[row_nr]:
            recovered_org_grad.append(norm_org)
            recovered_2nd_grad.append(norm_2nd)
        else:
            not_recovered_org_grad.append(norm_org)
            not_recovered_2nd_grad.append(norm_2nd)

    values = [
        adv_strength,
        count_lowest / H,
        H / correctly_classified,
        correctly_classified,
        count_lowest,
        count_highest,
        H,
        np.sum(recovered),
        np.mean(dist_adv_org)]
    results.append(values)
    values_str = delimiter.join([str(x) for x in values])
    print(values_str)


def compute():
    # file_name = 'gradients_confidence_org_adv.csv'
    # file_name = '2019-11-15-04-12-26-863148_cifar10_grad_stats.csv'
    # class_type = 'original'
    class_type = 'adv'
    header = [
        'adv strength',
        'agreement lowest gradient and max confidence',
        '% of adv (out of correctly classified)',
        'correctly classified',
        'norm lowest for ' + class_type,
        'norm highest for ' + class_type,
        'total values',
        'recovered number',
        'dist adv org']
    header_str = delimiter.join(header)
    print(header_str)
    results = []
    for adv_strength in [0.0001,
                         0.001,
                         # 0.002,
                         0.01,
                         0.04,
                         0.05,
                         0.1,
                         0.2,
                         1.0,
                         5.0,
                         10.0]:
        file_name = f'cw_{adv_strength}_gradients.csv'
        get_file_stats(file_name,
                       class_type=class_type,
                       results=results,
                       adv_strength=adv_strength)
    plot_results(np.array(results))

if __name__ == "__main__":
    compute()
