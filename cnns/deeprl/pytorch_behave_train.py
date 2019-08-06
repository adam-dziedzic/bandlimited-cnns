from cnns.nnlib.pytorch_experiments.main import train, test
from cnns.nnlib.pytorch_experiments.main import get_optimizer
from cnns.nnlib.pytorch_experiments.main import get_scheduler
from cnns.nnlib.pytorch_experiments.main import get_loss_function
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.deeprl.rollouts import get_rollouts_dataset
from cnns.deeprl.pytorch_model import load_model
import time
import numpy as np
import torch
import os
from cnns.nnlib.utils.general_utils import get_log_time

name = 'behave'


def save_model(model, train_loss, test_loss, args, epoch, returns=None):
    file_parts = [get_log_time(),
                  'env_name', args.env_name,
                  'rollouts', args.rollouts,
                  'epoch', epoch,
                  'train_loss', train_loss,
                  'test_loss', test_loss,
                  ]
    if returns is not None:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        file_parts.append('mean_return')
        file_parts.append(mean_return)
        file_parts.append('std_return')
        file_parts.append(std_return)
    models_dir = name + '_models'
    file_name = '_'.join([str(x) for x in file_parts])
    model_path = os.path.join(models_dir, file_name + '.model')
    torch.save(model.state_dict(), model_path)


def run(args):
    train_loader, test_loader, train_dataset, _ = get_rollouts_dataset(args)

    model = load_model(args)
    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(args=args, optimizer=optimizer)
    loss_function = get_loss_function(args)

    with open(args.log_file, 'a') as file:
        header = ['epoch', 'train_loss', 'test_loss',
                  'train_time', 'test_time']
        header_str = args.delimiter.join(header)
        file.write(args.get_str() + '\n')
        print(args.get_str())
        file.write(header_str + '\n')
        print(header_str)

        best_train_loss = float('inf')

        # Train the policy model on the initial data using behavioral cloning.
        for epoch in range(args.epochs):
            start_train = time.time()
            train_loss, _ = train(model=model,
                                  train_loader=train_loader,
                                  optimizer=optimizer,
                                  loss_function=loss_function,
                                  args=args,
                                  epoch=epoch)
            train_time = time.time() - start_train

            # scheduler step is based only on the train data, we do not use the
            # test data to schedule the decrease in the learning rate.
            scheduler.step(train_loss)

            start_test = time.time()
            test_loss, _ = test(
                model=model, test_loader=test_loader,
                loss_function=loss_function, args=args)
            test_time = time.time() - start_test

            with open(args.log_file, mode='a') as file:
                data = [epoch, train_loss, test_loss, train_time,
                        test_time]
                data_str = args.delimiter.join([str(x) for x in data])
                file.write(data_str + '\n')
                print(data_str)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                save_model(model=model,
                           train_loss=train_loss,
                           test_loss=test_loss,
                           epoch=epoch,
                           args=args)


if __name__ == "__main__":
    args = get_args()
    run(args=args)
