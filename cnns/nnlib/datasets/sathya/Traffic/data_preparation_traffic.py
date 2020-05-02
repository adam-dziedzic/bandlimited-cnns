import numpy as np
import os
import sys
import time

print("current working directory: ", os.getcwd())

# type = "_small"  # nothing i.e. "" normal or "_small" for small files
type = ""


def get_min_len(datasets):
    min_len = sys.maxsize  # get the minimum length of dataset for each class
    for dataset in datasets.values():
        if len(dataset) < min_len:
            min_len = len(dataset)
    return min_len


def set_dataset(datasets, file_name, counter, start_counter=0):
    start_time = time.time()
    csv_path = os.getcwd() + '/raw_data/' + file_name + '.txt'
    print('csv path: ', csv_path)

    dataset = np.genfromtxt(csv_path,
                            delimiter="\n",
                            missing_values='',
                            filling_values='inf',
                            skip_header=1,
                            skip_footer=1)
    for expression in ['-inf', '-Inf', 'inf', 'Inf']:
        dataset = np.delete(
            dataset, np.where(
                dataset == float(expression)))
    dataset = dataset[~np.isnan(dataset)]
    print("dataset class " + str(counter))
    print("max: ", dataset.max())
    print("min: ", dataset.min())
    print("mean: ", dataset.mean())
    print("len: ", len(dataset))
    # print("data1 head: ", data1.head())
    print("head: ", dataset[:10])
    class_number = counter - start_counter
    if counter in datasets.keys():
        datasets[class_number] = np.concatenate(
            (datasets[class_number], dataset), axis=0)
    else:
        datasets[class_number] = dataset
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('elapsed time to read a single data file: ',
          elapsed_time)


def generate_dataset(sample_size, datasets, train_rate, outlier_std_count):
    class_nr = len(datasets)
    # Only truncate the raw dataset sizes.
    set_min_len_manually = False
    if set_min_len_manually:
        min_len = 512 * 2000
    else:
        min_len = get_min_len(datasets=datasets)

    print("min_len of the dataset for a class: ", min_len)
    # make sure we have the same number of samples from each class
    for i in datasets.keys():
        datasets[i] = datasets[i][:min_len]
    del min_len

    def get_samples(array):
        with_step = True
        if with_step:
            # make more data by overlapping the signals
            step = max(sample_size // 4, 1)
            # step = 1
            samples = []
            # i - a start index for a sample
            for i in range(0, len(array) - sample_size, step):
                samples.append(array[i:i + sample_size])
            frame = np.array(samples)
        else:
            len_final = len(array)
            len_reminder = len_final % (sample_size * 2)
            len_final -= len_reminder
            array = array[:len_final]
            # cut off the data to the multiple of sample size
            # take sample_size value and create a row from them
            frame = array.reshape(-1, sample_size)
        # shuffle the data
        np.random.shuffle(frame)
        return frame

    # get 2 dimensional datasets
    for i in datasets.keys():
        datasets[i] = get_samples(datasets[i])

    # divide into train/test datasets
    min_len = get_min_len(datasets=datasets)

    stop_train_index = int(np.ceil(min_len * train_rate))

    train_arrays = {}
    test_arrays = {}

    for class_nr, dataset in datasets.items():
        train_arrays[class_nr] = dataset[:stop_train_index]
        test_arrays[class_nr] = dataset[stop_train_index:]

    # find the mean and std of the train data
    train_raw_arrays = []
    for array in train_arrays.values():
        train_raw_arrays.append(array)
    train_raw = np.concatenate(train_raw_arrays, axis=0)

    mean = train_raw.mean()
    print("train mean value: ", mean)

    std = train_raw.std()
    print("train std: ", std)

    def get_final_data(data, class_number, mean, std, type=''):
        # replace outliers with the mean value
        count_outliers = np.sum(np.abs(data - mean) > outlier_std_count * std)
        print(f"count_outliers {type} (for class: {class_number}): ",
              count_outliers)
        data[np.abs(data - mean) > outlier_std_count * std] = mean
        # normalize the data
        data = (data - mean) / std
        # create and add column with the class number
        class_column = np.full((len(data), 1), class_number)
        data = np.concatenate((class_column, data), axis=1)
        return data

    train_datasets = []
    for class_number, array in train_arrays.items():
        train_datasets.append(
            get_final_data(array,
                           class_number=class_number,
                           mean=mean,
                           std=std,
                           type='train'))
    data_train = np.concatenate(train_datasets, axis=0)
    del train_datasets

    test_datasets = []
    for class_number, array in test_arrays.items():
        test_datasets.append(
            get_final_data(array, class_number=class_number, mean=mean,
                           std=std, type='test'))
    data_test = np.concatenate(test_datasets, axis=0)
    del test_datasets

    dir_name = 'traffic_data/Traffic' + str(class_nr) + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    def write_data(data_set, file_name):
        with open(file_name, "w") as f:
            for row in data_set:
                # first row is a class number (starting from 0)
                f.write(str(int(row[0])))
                # then we have proper values starting from position 1 in each row
                for value in row[1:]:
                    f.write("," + str(value))
                f.write("\n")

    full_dir = dir_name + '/Traffic' + str(class_nr)
    write_data(data_train, full_dir + "_TRAIN")
    write_data(data_test, full_dir + "_TEST")

    print("normalized train mean (should be close to 0): ", data_train.mean())
    print("normalized train std: ", data_train.std())
    print("normalized test mean (should be close to 0): ", data_test.mean())
    print("normalized test std: ", data_test.std())


if __name__ == "__main__":
    # sample_sizes = [2, 4, ]  # 512
    sample_sizes = [512]
    # for sample_size in [2**x for x in range(10, 0, -1)]:
    for sample_size in sample_sizes:
        print("sample size: ", str(sample_size))
        train_rate = 0.5  # rate of training data, test data rate is 1 - train_rate
        outlier_std_count = 10
        file_names = ['streaming', 'video', 'data', 'base', 'data-video']
        datasets = dict()
        for counter, file_name in enumerate(file_names):
            set_dataset(datasets=datasets,
                        counter=counter,
                        file_name=file_name,
                        )
        generate_dataset(sample_size=sample_size,
                         datasets=datasets,
                         train_rate=train_rate,
                         outlier_std_count=outlier_std_count,
                         )
