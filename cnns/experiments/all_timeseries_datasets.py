import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print("current working directory: ", dir_path)

data_folder = "nnlib/TimeSeriesDatasets"
ucr_path = os.path.join(dir_path, os.pardir, data_folder)

flist = os.listdir(ucr_path)

flist = sorted(flist, key=lambda s: s.lower())
print("flist: ", flist)