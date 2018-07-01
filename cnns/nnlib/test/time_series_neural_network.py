# As usual, a bit of setup
from __future__ import print_function

from classifiers.cnn_time_series_1D import ThreeLayerConvNetTimeSeries
from fast_layers import *
from load_time_series import load_data
from solver import Solver

np.random.seed(231)


def reshapeTS(x):
    """
    Reshape the time-series data to have only a single dimension for
    channels and height. Move the time-series value to the width dimension.
    """
    return x.reshape(x.shape[0], 1, -1)

# load the data for time-series
dirname = "50words"
# dirname = "OSULeaf"
# dirname = "Coffee"
datasets = load_data(dirname)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

num_classes = len(np.unique(test_set_y))

num_train = 450
small_data = {
    'X_train': reshapeTS(train_set_x[:num_train]),
    'y_train': reshapeTS(train_set_y[:num_train]),
    'X_val': reshapeTS(valid_set_x),
    'y_val': reshapeTS(valid_set_y)
}

time_dimension = valid_set_x.shape[1]
print("time dimension: ", time_dimension)
model = ThreeLayerConvNetTimeSeries(input_dim=(1, 1, time_dimension),
                                    num_filters=32,
                                    filter_size=3,
                                    filter_channels=1,
                                    hidden_dim=500,
                                    num_classes=num_classes,
                                    weight_scale=1e-2,
                                    pad_convolution=2,
                                    stride_convolution=1)

solver = Solver(model, small_data,
                num_epochs=10000000, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                    'lr_decay': 0.7,
                },
                verbose=True, print_every=1)
solver.train()
