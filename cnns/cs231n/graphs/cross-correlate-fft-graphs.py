import matplotlib.patches as mpatches
from pandas import DataFrame

from cs231n.layers import *
from cs231n.layers import _ncc_c
from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import reshape_3d_rest, abs_error
from cs231n.utils.perf_timing import wrapper, timeitrep

np.random.seed(231)

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = train_set_x[0]
filter_size = 4
full_filter = train_set_x[1]
filters = full_filter[:filter_size]

