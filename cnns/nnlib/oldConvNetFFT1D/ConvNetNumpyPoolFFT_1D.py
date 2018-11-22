from nnlib.classifiers.cnn_numpy_pool_fft_1D import ThreeLayerConvNetNumpyPoolFFT1D
from nnlib.data_utils import get_CIFAR10_data
from nnlib.solver import Solver
from nnlib.utils.general_utils import get_log_time

data = get_CIFAR10_data(cifar10_dir='datasets/cifar-10-batches-py')
for k, v in data.items():
    print('%s: ' % k, v.shape)

num_train = 49000
num_valid = 1000
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

epoch_log = "numpy_conv_fft_pool_epoch_log_" + get_log_time() + ".csv"
loss_log = "numpy_conv_fft_pool_loss_log_" + get_log_time() + ".csv"

model = ThreeLayerConvNetNumpyPoolFFT1D(weight_scale=1e-2)
solver = Solver(model, small_data,
                num_epochs=1000000, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1,
                epoch_log=epoch_log,
                loss_log=loss_log
                )
solver.train()

print("number of epochs: ", solver.num_epochs)
print("loss history: ", solver.loss_history)
