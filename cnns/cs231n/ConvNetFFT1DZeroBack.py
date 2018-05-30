import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
from pandas import DataFrame

from cs231n.classifiers.cnn_fft_1D import ThreeLayerConvNetFFT1D
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver
from cs231n.utils.general_utils import *


class Result(object):
    def __init__(self, data):
        self.data = data


current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

data = get_CIFAR10_data(cifar10_dir='datasets/cifar-10-batches-py')
for k, v in data.items():
    print('%s: ' % k, v.shape)

print("number of training examples: ", len(data['X_train']))
print("number of training examples: ", data['X_train'].shape)

#num_train = 49000
#num_valid = 1000
# num_train = 10
# num_valid = 10
num_train = 3000
num_valid = 300

small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'][:num_valid],
    'y_val': data['y_val'][:num_valid],
}
#print("X_train: ", small_data['X_train'].shape)
#print("y_train: ", data['y_train'])

small_data['X_train'] = small_data['X_train'].reshape(
    small_data['X_train'].shape[0], small_data['X_train'].shape[1], -1)
#print("x_train shape: ", small_data['X_train'].shape)

small_data['X_val'] = small_data['X_val'].reshape(
    small_data['X_val'].shape[0], small_data['X_val'].shape[1], -1)
#print("x_val shape: ", small_data['X_val'].shape)

print_every = num_train
num_epochs = 1000
energy_rates = [None]
# num_epochs = 2
#energy_rates = [1.0, 0.999]
losses_rate = []

epoch_log = "fft_epoch_log_" + get_log_time() + ".csv"
loss_log = "fft_loss_log_" + get_log_time() + ".csv"

for energy_rate in energy_rates:
    print("energy rate: ", energy_rate)
    start = time.time()
    model = ThreeLayerConvNetFFT1D(weight_scale=1e-2, energy_rate_convolution=energy_rate, index_back=None)
    solver = Solver(model, small_data,
                    num_epochs=num_epochs, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=print_every,
                    epoch_log=epoch_log,
                    loss_log=loss_log
                    )
    solver.train()
    elapsed_time = time.time() - start
    losses_rate.append((solver.train_acc_history, solver.val_acc_history, energy_rate, solver.loss_history,
                        elapsed_time))
    print("energy rate: ", solver.num_epochs)
    print("loss history: ", solver.loss_history)
    print("train acc: ", solver.train_acc_history)
    print("val acc: ", solver.val_acc_history)
    print("elapsed time: ", elapsed_time)

    data = losses_rate
    result = Result(data)
    pickle_name = "results/" + current_file_name + "-" + get_log_time() + ".pkl"
    print("pickle name: ", pickle_name)
    save_object(result, pickle_name)

    # get current file name

    # import matplotlib.pyplot as plt
    # import os
    #
    # epochs = 5
    # energy1 = [1, 2, 3, 4, 5]
    # energy2 = [1, 2, 1, 3, 4]
    # energy3 = [0, 0, 2, 3, 10]
    # energies = [(energy1, 0.99), (energy2, 0.90), (energy3, 0.80)]

    epochs = [epoch for epoch in range(num_epochs)]
    fig, ax = plt.subplots()
    for train_acc, val_acc, rate, _, _ in losses_rate:
        train_label = "train-" + str(rate)
        ax.plot([x for x in range(len(train_acc))], train_acc, label=train_label)
        val_label = "validation-" + str(rate)
        ax.plot([x for x in range(len(val_acc))], val_acc, label=val_label)

    ax.legend()
    plt.xticks(epochs)
    plt.title('Compare loss for fft convolution with different preserved energy rates')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Validation accuracy')
    plt.savefig("graphs/train-val-" + current_file_name + "-" + get_log_time() + ".png")
    plt.gcf().subplots_adjust(bottom=0.10)
    plt.savefig("graphs/train-val-" + current_file_name + "-" + get_log_time() + ".pdf")
    #plt.show()

    fig, ax = plt.subplots()
    for _, _, rate, losses, _ in losses_rate:
        ax.plot([x for x in range(len(losses))], losses, label=str(rate))

    ax.legend()
    plt.xticks(epochs)
    plt.title('Compare loss for fft convolution with different preserved energy rates')
    plt.xlabel('Time step')
    plt.ylabel('Train loss')
    plt.savefig("graphs/losses-" + current_file_name + "-" + get_log_time() + ".png")
    plt.gcf().subplots_adjust(bottom=0.10)
    plt.savefig("graphs/losses-" + current_file_name + "-" + get_log_time() + ".pdf")
    #plt.show()

    # import matplotlib.patches as mpatches
    # import matplotlib.pyplot as plt
    # from pandas import DataFrame
    # current_file_name = ""
    #
    # losses_rate = [([], [], 1.0, [], 3.0), ([], [], 0.9, [], 7.9)]

    trates = []
    ttimes = []
    for _, _, rate, _, timing in losses_rate:
        print(timing)
        trates.append(rate)
        ttimes.append(timing)

    # print("timings: ", ttimes)
    # print("trates: ", trates)
    df_input = {'rates': trates, 'ttimes': ttimes}
    df = DataFrame(data=df_input)
    df=df.astype(float)
    #print("df: ", df)

    fig = plt.figure()  # create matplot figure
    ax = fig.add_subplot(111)  # create matplotlib axes

    width = 1.0

    df.ttimes.plot(kind='bar', color='red', ax=ax)

    left_ylabel = "Execution time (sec)"
    ax.set_ylabel(left_ylabel, color="red")
    ax.tick_params('y', colors='red')

    plt.title("Execution time (sec) for each preserved energy rate")
    ax.set_xticklabels(trates)
    ax.set_xlabel("Preserved energy rate")
    red_patch = mpatches.Patch(color='red', label=left_ylabel)

    plt.legend(handles=[red_patch], loc='upper right', ncol=1,
               borderaxespad=0.0)
    plt.savefig("graphs/timing-" + current_file_name + "-" + get_log_time() + ".png")
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.savefig("graphs/timing-" + current_file_name + "-" + get_log_time() + ".pdf")
    #plt.show()
