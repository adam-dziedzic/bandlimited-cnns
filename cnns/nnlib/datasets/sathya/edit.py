import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns


type = "_small"  # "" normal or "_small" small
sample_size = 500  # how many values in a single sample collected
train_rate = 0.5
outlier_std_count = 3

csv_path1 = "test_one_wifi_1" + type + ".txt"
data1 = pd.read_csv(csv_path1, header=None)
print("data1 values: ", data1.values)
data1 = np.array(data1.values).squeeze()
print("data1 max: ", data1.max())
print("data1 mean: ", data1.mean())
# print("data1 head: ", data1.head())
print("length of data1: ", len(data1))

stop_train_index = int(len(data1) * train_rate)

data1_train = data1[:stop_train_index]
data1_test = data1[stop_train_index:]

csv_path2 = "test_one_wifi_2" + type + ".txt"
data2 = pd.read_csv(csv_path2, header=None)
print("data2 head: ", data2.head())
data2 = np.array(data2.values).squeeze()
print("data2 min: ", data2.min())
print("data2 mean: ", data2.mean())
stop_train_index = int(len(data2) * train_rate)

data2_train = data2[:stop_train_index]
data2_test = data2[stop_train_index:]

# find the mean and std of the train data
train_arrays = [data1_train, data2_train]
train_raw = np.concatenate(train_arrays)

mean = train_raw.mean()
print("mean value: ", mean)

std = train_raw.std()
print("std: ", std)

def mean_out_outliers(data, mean, std):
    count_outliers = 0
    for index, value in enumerate(data):
        # replace an outlier with zero, outside 3 standard deviations
        if abs(value - mean) > outlier_std_count * std:
            data[index] = mean
            count_outliers += 1
    return count_outliers


def get_timeseries_frame(frame, class_number, mean, std):
    # replace outliers with the mean value
    count_outliers = sum(abs(frame - mean) > outlier_std_count * std)
    print("count_outliers: ", count_outliers)
    frame[abs(frame - mean) > outlier_std_count * std] = mean
    # normalize the data
    frame = (frame - mean) / std
    # take sample_size value and create a row from them
    frame = frame.reshape(-1, sample_size)
    # add column with the class number
    class_column = np.full((len(frame), 1), class_number)
    frame = np.concatenate((class_column, frame), axis=1)
    return frame

data2_train = get_timeseries_frame(data2_train, class_number=1, mean=mean,
                                   std=std)
print("data2_train mean: ", data2_train[:,1:].mean(axis=1))

data1_train = get_timeseries_frame(data1_train, class_number=0, mean=mean,
                                   std=std)
print("data1_train mean: ", data1_train[:,1:].mean(axis=1))



with open("test_one_wifi_1_class_1000_vals.txt", "w"):
    pass
