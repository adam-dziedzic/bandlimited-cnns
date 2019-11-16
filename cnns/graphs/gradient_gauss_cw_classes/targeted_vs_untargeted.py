import numpy as np
import os
import pandas as pd

delimiter = ';'


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
    col_name_from_col_nr = data_all[col_nr - 1][0]
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


def compute():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # attack_type = 'targeted'
    attack_type = 'untargeted'
    if attack_type == 'targeted':
        delimiter = ';'
        file_name = '2019-11-15-12-16-53-002162_cifar10_grad_stats.csv'
    elif attack_type == 'untargeted':
        delimiter = ','
        file_name = '2019-11-15-04-12-26-863148_cifar10_grad_stata_cw_classes.csv'
    else:
        raise Exception(f'Unknown attack type: {attack_type}')

    data_path = os.path.join(dir_path, file_name)
    data_all = pd.read_csv(data_path, header=None, sep=delimiter)
    print('shape of data all: ', data_all.shape)

    # limit = None
    limit = 169
    if limit:
        data_all = data_all.iloc[:limit]

    H, W = data_all.shape
    print('row number: ', H)
    print(data_all.head(5))

    # class_type = 'original'
    class_type = 'adv'

    col_names = []
    for class_nr in range(10):
        col_names += ['l2_norm_' + class_type + '_class_' + str(class_nr)]
    norm_vals = get_col_vals(data_all=data_all, col_names=col_names)

    target_classes = get_col_val(data_all=data_all,
                                 col_name=class_type + '_class',
                                 dtype=np.int)

    recovered_nr = get_col_val(data_all=data_all,
                               col_name='is_gauss_recovered',
                               dtype=np.bool)

    dist_adv_org = get_col_val(data_all=data_all,
                               col_name='z_l2_dist_adv_org_image',
                               dtype=np.float)


    class_type = 'org' if class_type == 'original' else 'adv'
    col_names = []
    for class_nr in range(10):
        col_names += [class_type + '_confidence_class_' + str(class_nr)]
    conf_vals = get_col_vals(data_all=data_all, col_names=col_names)

    count_agree = 0
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
        class_max = np.argmin(norms)
        if class_max == class_nr:
            count_agree += 1

    header = ['norm lowest for ' + class_type,
              'total values',
              'recovered number',
              'dist adv org']
    header_str = delimiter.join(header)
    print(header_str)
    values = [count_agree,
              H,
              np.sum(recovered_nr),
              np.mean(dist_adv_org)]
    values_str = delimiter.join([str(x) for x in values])
    print(values_str)

    col_names = ['l2_norm_gauss_correct',
                 'l2_norm_gauss_adv',
                 'l2_norm_gauss_zero',
                 'l2_norm_original_correct',
                 'l2_norm_original_adv',
                 'l2_norm_original_class_0',
                 'l2_norm_adv_adv',
                 'l2_norm_adv_correct',
                 'l2_norm_adv_class_0',
                 'adv_confidence', 'original_confidence',
                 'z_l2_dist_adv_org_image']
    col_vals = get_col_vals(data_all=data_all, col_names=col_names)

    params = [0.001, 0.002, 0.004, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
              0.08, 0.1, 0.5, 1.0]

    recovered_nr = get_col_val(data_all=data_all, col_name='is_gauss_recovered',
                               dtype=np.bool)

    header = ["i", "param"]
    for col_name in col_names:
        for key in stats.keys():
            header += [col_name + "-" + key]
    header += ['nr_recovered']
    header_str = delimiter.join(header)
    print(header_str)

    step = 939
    start_index = -step
    end_index = 0


    for i, param in enumerate(params):
        start_index += step
        end_index += step

        info = [i, param]
        for col_val in col_vals:
            col_stats = get_stats(col_val[start_index:end_index])
            info += col_stats.values()

        nr_recovered = np.sum(recovered_nr[start_index:end_index])
        info += [nr_recovered]

        info_str = delimiter.join([str(x) for x in info])
        print(info_str)


if __name__ == "__main__":
    compute()
