# import pandas as pd
import numpy as np
import os
import sys

# from sklearn import preprocessing
# import seaborn as sns

print("current working directory: ", os.getcwd())

# type = "_small"  # nothing i.e. "" normal or "_small" for small files
type = ""
# sample_size: 1000, 500, 250, 32, 64
# sample_size = 192  # 500 for small data # how many values in a single sample collected
# sample_size = 2048

# for sample_size in [2**x for x in range(12, 0, -1)]:
for sample_size in [512]:
    print("sample size: ", str(sample_size))
    train_rate = 0.5  # rate of training data, test data rate is 1 - train_rate
    outlier_std_count = 10
    # class_counter = 5
    # class_counter = 6
    # prefix="test_one_wifi_"
    # prefix="wifi6_data/"
    # suffix="_wifi_165"

    class_counter = 2
    class_number = None
    # prefix = "ML_"
    prefix = 'CaseB'
    suffix = "WiFi_"
    los_type = 'LOS'  # LOS or NLOS
    distance = 10

    datasets = []
    min_len = sys.maxsize  # get the minimum length of dataset for each class

    for counter in range(class_counter):
        # csv_path = prefix + los_type + '/' + str(
        #     distance) + 'F_' + los_type + '/Test_165_' + str(
        #     distance) + 'F_' + str(
        #     counter) + suffix + los_type.lower() + type + ".txt"
        # csv_path = prefix + '/Test_165_' + prefix + '_1WiFi_' + los_type.lower() + '.txt'
        # csv_path = 'All/raw_files/all_data'
        csv_path = 'All/one_two_wifi/' + str(counter + 1) + '_wifi'
        csv_path = 'All/one_two_wifi_2/' + str(counter + 1) + '_wifi_2'

        print('csv path: ', csv_path)

        # data1 = pd.read_csv(csv_path1, header=None)
        # # print("data1 values: ", data1.values)
        # data1 = np.array(data1.values).squeeze()

        dataset = np.genfromtxt(csv_path, delimiter="\n")
        for expression in ['-inf', '-Inf', 'inf', 'Inf']:
            dataset = np.delete(dataset, np.where(dataset == float(expression)))
        dataset = dataset[~np.isnan(dataset)]
        print("dataset class " + str(counter))
        print("max: ", dataset.max())
        print("min: ", dataset.min())
        print("mean: ", dataset.mean())
        print("len: ", len(dataset))
        # print("data1 head: ", data1.head())
        print("head: ", dataset[:10])
        if len(dataset) < min_len:
            min_len = len(dataset)
        datasets.append(dataset)

    print("min_len of the dataset for a class: ", min_len)
    # make sure we have the same number of samples from each class
    for i in range(len(datasets)):
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
    for i in range(len(datasets)):
        datasets[i] = get_samples(datasets[i])

    # divide into train/test datasets
    stop_train_index = int(len(datasets[0]) * train_rate)

    train_arrays = []
    test_arrays = []

    for dataset in datasets:
        train_arrays.append(dataset[:stop_train_index])
        test_arrays.append(dataset[stop_train_index:])

    # find the mean and std of the train data
    train_raw = np.concatenate(train_arrays, axis=0)

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
    for i, array in enumerate(train_arrays):
        if class_number is None:
            class_number = i
        train_datasets.append(
            get_final_data(array, class_number=class_number, mean=mean,
                           std=std, type='train'))
    data_train = np.concatenate(train_datasets, axis=0)
    del train_datasets

    test_datasets = []
    for i, array in enumerate(test_arrays):
        if class_number is None:
            class_number = i
        test_datasets.append(
            get_final_data(array, class_number=class_number, mean=mean,
                           std=std, type='test'))
    data_test = np.concatenate(test_datasets, axis=0)
    del test_datasets

    # print("data train dims: ", data_train.shape)
    # np.savetxt("WIFI_TRAIN", data_train, delimiter=",")
    sample_size = str(sample_size)
    # dataset_name = "WIFI_class_" + str(class_counter)
    dataset_name = prefix + los_type + '/' + str(
        distance) + 'F_' + los_type + '/'
    dir_name = str(class_counter) + '_classes_WiFi'
    # full_dir = dataset_name + "/" + dir_name
    # full_dir = prefix + '/' + prefix + '_' + los_type.lower()
    # full_dir = 'All/raw_files/all_data'
    # full_dir = 'All/one_two_wifi/2_classes_WiFi'
    full_dir = 'All/one_two_wifi_2/2_classes_WiFi'

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


    write_data(data_train, full_dir + "_TRAIN")
    write_data(data_test, full_dir + "_TEST")

    print("normalized train mean (should be close to 0): ", data_train.mean())
    print("normalized train std: ", data_train.std())
    print("normalized test mean (should be close to 0): ", data_test.mean())
    print("normalized test std: ", data_test.std())
