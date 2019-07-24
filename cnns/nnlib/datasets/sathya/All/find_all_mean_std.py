import numpy as np

all = np.genfromtxt('raw_files/all_data', delimiter='\n')

# dataset = np.genfromtxt(csv_path, delimiter="\n")
for expression in ['-inf', '-Inf', 'inf', 'Inf', 'nan']:
    dataset = np.delete(dataset, np.where(dataset == float(expression)))

dataset[~np.isnan(dataset)].mean()
# -37.09453644990092
dataset[~np.isnan(dataset)].std()
# 2.7407070318624496
