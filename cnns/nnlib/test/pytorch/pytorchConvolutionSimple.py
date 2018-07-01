# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import reshape_3d_rest

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:2") # Uncomment this to run on GPU

np.random.seed(231)

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# x = train_set_x[0]
x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
filter_size = 4
# full_filter = train_set_x[1]
# filters = full_filter[:filter_size].copy()
# filters = np.random.randn(filter_size)
filters = np.array([1.0, 2, 0, 1])

repetitions = 20
exec_number = 1

b = np.array([0])

stride = 1

mode = "full"
if mode == "valid":
    padding = 0
elif mode == "full":
    padding = len(filters) - 1
conv_param = {'stride': stride, 'pad': padding}

timings = []
errors = []

fraction = 0.99

x = reshape_3d_rest(x)
filters = reshape_3d_rest(filters)

xtorch = torch.from_numpy(x)
filtertorch = torch.from_numpy(filters)

print(xtorch)

# conv1d = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=filter_size, stride=stride, padding=padding,
#                          bias=False)
# conv1d.forward(input=xtorch)
result = F.conv1d(input=xtorch, weight=filtertorch, bias=None, stride=stride, padding=padding, dilation=1, groups=1)
result_pytorch = result.numpy()

print("result pytorch: ", result_pytorch)
