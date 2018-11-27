import pandas as pd
import numpy as np

type = "_small" # "" normal or "_small" small

csv_path1 = "test_one_wifi_1" + type + ".txt"
data1 = pd.read_csv(csv_path1, header=None)
print("data1 head: ", data1.head())

# find the mean and std of the train data

with open("test_one_wifi_1_class_1000_vals.txt", "w"):
    array = np.array(1000)
    for index, row in data1.iterrows():
        array.append(row)
        if len(array) == 1000:

            array.empty

csv_path2 = "test_one_wifi_2" + type + ".txt"
data2 = pd.read_csv(csv_path2, header=None)
print("data2 head: ", data2.head())