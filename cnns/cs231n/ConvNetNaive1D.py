from cs231n.classifiers.cnn_naive_1D import ThreeLayerConvNetNaive1D
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

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


model = ThreeLayerConvNetNaive1D(weight_scale=1e-2)

solver = Solver(model, small_data,
                num_epochs=15, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
solver.train()
