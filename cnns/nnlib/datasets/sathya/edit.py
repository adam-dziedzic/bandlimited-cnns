import pandas as pd
import numpy as np
# from sklearn import preprocessing
# import seaborn as sns

# type = "_small"  # nothing i.e. "" normal or "_small" for small files
type = ""
sample_size = 1000 # 500 for small data # how many values in a single sample collected
train_rate = 0.5  # rate of training data, test data rate is 1 - train_rate
outlier_std_count = 4

csv_path1 = "test_one_wifi_1" + type + ".txt"
data1 = pd.read_csv(csv_path1, header=None)
# print("data1 values: ", data1.values)
data1 = np.array(data1.values).squeeze()
print("data1 max: ", data1.max())
print("data1 min: ", data1.min())
print("data1 mean: ", data1.mean())
# print("data1 head: ", data1.head())

stop_train_index = int(len(data1) * train_rate)

csv_path2 = "test_one_wifi_2" + type + ".txt"
data2 = pd.read_csv(csv_path2, header=None)
# print("data2 head: ", data2.head())
data2 = np.array(data2.values).squeeze()
print("data2 min: ", data2.min())
print("data2 max: ", data2.max())
print("data2 mean: ", data2.mean())

len1 = len(data1)
len2 = len(data2)

print("length of data1: ", len1)
print("length of data2: ", len2)

len_final = min(len1, len2)
len_reminder = len_final % (sample_size*2)
len_final -= len_reminder

print("final length of datasets:", len_final)
data1 = data1[:len_final]
data2 = data2[:len_final]

# divide into train/test parts
stop_train_index = int(len_final * train_rate)
data1_train = data1[:stop_train_index]
data1_test = data1[stop_train_index:]

data2_train = data2[:stop_train_index]
data2_test = data2[stop_train_index:]

# find the mean and std of the train data
train_arrays = [data1_train, data2_train]
train_raw = np.concatenate(train_arrays)

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


def get_timeseries_frame(frame, class_number, mean, std):
    # replace outliers with the mean value
    count_outliers = sum(abs(frame - mean) > outlier_std_count * std)
    print("count_outliers: ", count_outliers)
    frame[abs(frame - mean) > outlier_std_count * std] = mean
    # normalize the data
    frame = (frame - mean) / std
    # cut off the data to the multiple of sample size
    # take sample_size value and create a row from them
    frame = frame.reshape(-1, sample_size)
    # add column with the class number
    class_column = np.full((len(frame), 1), class_number)
    frame = np.concatenate((class_column, frame), axis=1)
    return frame

def write_data(data_set, file_name):
    with open(file_name, "w") as f:
        for row in data_set:
            # first row is a class number (starting from 0)
            f.write(str(int(row[0])))
            # then we have proper values starting from position 1 in each row
            for value in row[1:]:
                f.write("," + str(value))
            f.write("\n")

data1_train = get_timeseries_frame(data1_train, class_number=0, mean=mean,
                                   std=std)
# print("data1_train mean: ", data1_train[:,1:].mean(axis=1))

data2_train = get_timeseries_frame(data2_train, class_number=1, mean=mean,
                                   std=std)
# print("data2_train mean: ", data2_train[:,1:].mean(axis=1))

data_train = np.concatenate((data1_train, data2_train), axis=0)
# print("data train dims: ", data_train.shape)
# np.savetxt("WIFI_TRAIN", data_train, delimiter=",")
write_data(data_train, "WIFI_TRAIN")

data1_test = get_timeseries_frame(data1_test, class_number=0, mean=mean,
                                   std=std)
data2_test = get_timeseries_frame(data2_test, class_number=1, mean=mean,
                                   std=std)
data_test = np.concatenate((data1_test, data2_test), axis=0)
write_data(data_test, "WIFI_TEST")




