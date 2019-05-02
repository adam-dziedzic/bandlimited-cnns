import os
ucr_data_folder = "TimeSeriesDatasets"
# ucr_path = os.path.join(dir_path, os.pardir, data_folder)
ucr_path = os.path.join(os.pardir, ucr_data_folder)
print(sorted(os.listdir(ucr_path)))