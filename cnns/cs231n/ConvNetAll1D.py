from cs231n.classifiers.cnn_numpy_1D import ThreeLayerConvNetNumpy1D
from cs231n.classifiers.cnn_fft_1D import ThreeLayerConvNetFFT1D
from cs231n.classifiers.cnn_naive_1D import ThreeLayerConvNetNaive1D
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver
from cs231n.utils.general_utils import *

import os
import matplotlib.pyplot as plt

data = get_CIFAR10_data(cifar10_dir='datasets/cifar-10-batches-py')
for k, v in data.items():
    print('%s: ' % k, v.shape)

num_train = 10
num_valid = 10
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'][:num_valid],
    'y_val': data['y_val'][:num_valid],
}
print("X_train: ", small_data['X_train'].shape)
print("y_train: ", data['y_train'])

small_data['X_train'] = small_data['X_train'].reshape(
    small_data['X_train'].shape[0], small_data['X_train'].shape[1], -1)
print("x_train shape: ", small_data['X_train'].shape)

small_data['X_val'] = small_data['X_val'].reshape(
    small_data['X_val'].shape[0], small_data['X_val'].shape[1], -1)
print("x_val shape: ", small_data['X_val'].shape)

epochs = 15

model = ThreeLayerConvNetNaive1D(weight_scale=1e-2)
solver = Solver(model, small_data,
                num_epochs=epochs, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()

print("number of epochs naive: ", solver.num_epochs)
print("loss history naive: ", solver.loss_history)
naive_loss = solver.loss_history

model = ThreeLayerConvNetNumpy1D(weight_scale=1e-2)
solver = Solver(model, small_data,
                num_epochs=epochs, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()

print("number of epochs numpy: ", solver.num_epochs)
print("loss history numpy: ", solver.loss_history)
numpy_loss = solver.loss_history

model = ThreeLayerConvNetFFT1D(weight_scale=1e-2)
solver = Solver(model, small_data,
                num_epochs=epochs, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()

print("number of epochs fft: ", solver.num_epochs)
print("loss history fft: ", solver.loss_history)
fft_loss = solver.loss_history


class Result(object):
    def __init__(self, data):
        self.data = data


# get current file name
cwd = os.getcwd()
current_file_name = cwd.split('/')[-1]

data = [epochs, naive_loss, numpy_loss, fft_loss]
result = Result(data)
save_object(result, "results/" + current_file_name + "-" + get_log_time() + ".pkl")

# import matplotlib.pyplot as plt
# epochs = 2
# naive_loss = [2.239094894705599, 0.7135400620262297]
# numpy_loss = [2.3355538515222145, 0.8984360804725811]
# fft_loss = [2.701234428271352, 1.5940949318593085]

epochs = [epoch for epoch in range(epochs)]
fig, ax = plt.subplots()
ax.plot(epochs, naive_loss, 'g^', label="naive")
ax.plot(epochs, numpy_loss, 'bs', label="numpy")
ax.plot(epochs, fft_loss, 'r--', label="fft")
ax.legend()
plt.xticks(epochs)
plt.title('Compare loss for naive, numpy and fft based convolution')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.savefig("graphs/" + current_file_name + "-" + get_log_time() + ".png")
plt.gcf().subplots_adjust(bottom=0.10)
plt.savefig("graphs/" + current_file_name + "-" + get_log_time() + ".pdf")
plt.show()
