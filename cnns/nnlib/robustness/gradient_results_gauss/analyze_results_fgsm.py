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
    'avg': np.average,
    'median': np.median,
    'std': np.std,
    'min': np.min,
    'max': np.max,
}


def get_stats(vals):
    results = {}
    for key, op in stats.items():
        results[key] = op(vals)
    return results


def get_column(data_all, col_nr, col_name, dtype=np.float):
    col_name_from_col_nr = data_all[col_nr - 1][0]
    print('col name: ', col_name_from_col_nr)
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
    file_name = "2019-11-07-20-58-55-215404_imagenet_grad_stats.csv"
    data_path = os.path.join(dir_path, file_name)
    data_all = pd.read_csv(data_path, header=None, sep=';')
    print('shape of data all: ', data_all.shape)
    H, W = data_all.shape
    print('row number: ', H)
    step = 895
    print('expected runs: ', H / step)
    print(data_all.head(5))

    noise_sigmas = [0.0000000001]
    noise_sigmas += [x / 1000 for x in range(1, 10)]
    noise_sigmas += [x / 100 for x in range(1, 10)]
    noise_sigmas += [x / 10 for x in range(1, 11)]

    fgsm_

    # check_image_indexes(data_all=data_all, step=step)

    col_names = ['l2_norm_gauss_correct', 'l2_norm_original_correct']
    col_vals = get_col_vals(data_all=data_all, col_names=col_names)

    recovered_nr = get_col_val(data_all=data_all, col_name='is_gauss_recovered',
                               dtype=np.bool)

    header = ['i', 'sigma']
    for _ in col_names:
        header += stats.keys()
    header += ['nr_recovered']
    header_str = delimiter.join(header)
    print(header_str)

    start_index = -step
    end_index = 0

    gauss_avg = []
    org_avg = []

    for i, sigma in enumerate(noise_sigmas):
        start_index += step
        end_index += step

        info = [i, sigma]
        for col_val in col_vals:
            col_stats = get_stats(col_val[start_index:end_index])
            info += col_stats.values()

        nr_recovered = np.sum(recovered_nr[start_index:end_index])
        info += [nr_recovered]

        info_str = delimiter.join([str(x) for x in info])
        print(info_str)


if __name__ == "__main__":
    compute()
