import pandas as pd
import numpy as np
import os

# from sklearn import preprocessing
# import seaborn as sns

# type = "_small"  # nothing i.e. "" normal or "_small" for small files
type = ""
# sample_size: 1000, 500, 250, 32, 64
sample_size = 256  # 500 for small data # how many values in a single sample collected
train_rate = 0.5  # rate of training data, test data rate is 1 - train_rate
outlier_std_count = 4

csv_path1 = "test_one_wifi_1" + type + ".txt"

# data1 = pd.read_csv(csv_path1, header=None)
# # print("data1 values: ", data1.values)
# data1 = np.array(data1.values).squeeze()

data1 = np.genfromtxt(csv_path1, delimiter="\n")

print("data1 max: ", data1.max())
print("data1 min: ", data1.min())
print("data1 mean: ", data1.mean())
# print("data1 head: ", data1.head())
print("data head: ", data1[:10])

stop_train_index = int(len(data1) * train_rate)

csv_path2 = "test_one_wifi_2" + type + ".txt"

# data2 = pd.read_csv(csv_path2, header=None)
# # print("data2 head: ", data2.head())
# data2 = np.array(data2.values).squeeze()

data2 = np.genfromtxt(csv_path2, delimiter="\n")

print("data2 min: ", data2.min())
print("data2 max: ", data2.max())
print("data2 mean: ", data2.mean())
print("data head: ", data2[:10])

len1 = len(data1)
len2 = len(data2)

print("length of data1: ", len1)
print("length of data2: ", len2)

len_final = min(len1, len2)

print("final length of datasets:", len_final)
data1 = data1[:len_final]
data2 = data2[:len_final]


def get_samples(array):
    with_step = True
    if with_step:
        # make more data by overlapping the signals
        step = sample_size // 4
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


data1 = get_samples(data1)
data2 = get_samples(data2)

assert len(data1) == len(data2)
len_final = len(data1)

# divide into train/test parts
stop_train_index = int(len_final * train_rate)
data1_train = data1[:stop_train_index]
data1_test = data1[stop_train_index:]

data2_train = data2[:stop_train_index]
data2_test = data2[stop_train_index:]

# find the mean and std of the train data
train_arrays = [data1_train, data2_train]
train_raw = np.concatenate(train_arrays, axis=0)

mean = train_raw.mean()
print("train mean value: ", mean)

std = train_raw.std()
print("train std: ", std)


def mean_out_outliers(data, mean, std):
    count_outliers = 0
    for index, value in enumerate(data):
        # replace an outlier with zero, outside 3 standard deviations
        if abs(value - mean) > outlier_std_count * std:
            data[index] = mean
            count_outliers += 1
    return count_outliers


def get_final_data(data, class_number, mean, std):
    # replace outliers with the mean value
    count_outliers = np.sum(np.abs(data - mean) > outlier_std_count * std)
    print(f"count_outliers (for class: {class_number}): ", count_outliers)
    data[np.abs(data - mean) > outlier_std_count * std] = mean
    # normalize the data
    data = (data - mean) / std
    # create and add column with the class number
    class_column = np.full((len(data), 1), class_number)
    data = np.concatenate((class_column, data), axis=1)
    return data


def write_data(data_set, file_name):
    with open(file_name, "w") as f:
        for row in data_set:
            # first row is a class number (starting from 0)
            f.write(str(int(row[0])))
            # then we have proper values starting from position 1 in each row
            for value in row[1:]:
                f.write("," + str(value))
            f.write("\n")


data1_train = get_final_data(data1_train, class_number=0, mean=mean,
                             std=std)
# print("data1_train mean: ", data1_train[:,1:].mean(axis=1))

data2_train = get_final_data(data2_train, class_number=1, mean=mean,
                             std=std)
# print("data2_train mean: ", data2_train[:,1:].mean(axis=1))

data_train = np.concatenate((data1_train, data2_train), axis=0)

data1_test = get_final_data(data1_test, class_number=0, mean=mean,
                            std=std)
data2_test = get_final_data(data2_test, class_number=1, mean=mean,
                            std=std)
data_test = np.concatenate((data1_test, data2_test), axis=0)

# print("data train dims: ", data_train.shape)
# np.savetxt("WIFI_TRAIN", data_train, delimiter=",")
sample_size = str(sample_size)
dataset_name = "WIFI2"
dir_name = dataset_name + "-" + sample_size
full_dir = dir_name + "/" + dir_name

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

write_data(data_train, full_dir + "_TRAIN")
write_data(data_test, full_dir + "_TEST")
