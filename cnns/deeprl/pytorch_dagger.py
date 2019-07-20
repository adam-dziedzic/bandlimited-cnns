from cnns.nnlib.pytorch_experiments.main import train, test
from cnns.nnlib.pytorch_experiments.main import get_optimizer
from cnns.nnlib.pytorch_experiments.main import get_scheduler
from cnns.nnlib.pytorch_experiments.main import get_loss_function
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.deeprl.rollouts import get_rollouts_dataset
from cnns.deeprl.pytorch_model import load_model
from cnns.deeprl.models import run_model
from cnns.deeprl.pytorch_model import pytorch_policy_fn
from cnns.deeprl.load_policy import load_policy
from torch.utils.data import DataLoader
from cnns.nnlib.datasets.deeprl.rollouts import set_kwargs
import time


def run(args):
    train_loader, test_loader, train_dataset, _ = get_rollouts_dataset(args)

    model = load_model(args)
    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(args=args, optimizer=optimizer)
    loss_function = get_loss_function(args)

    with open(args.log_file, 'a') as file:
        header = ['dagger_iter', 'epoch', 'train_loss', 'test_loss',
                  'train_time', 'test_time']
        header_str = args.delimiter.join(header)
        file.write(header_str + '\n')
        print(header_str)

    for dagger_iter in range(args.dagger_iterations):
        # 1. train the policy model on the initial data.
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
                data = [dagger_iter, epoch, train_loss, test_loss, train_time,
                        test_time]
                data_str = args.delimiter.join([str(x) for x in data])
                file.write(data_str + '\n')
                print(data_str)

        # 2. Run the learned model to get new observations.
        # 3. At the same time, run the expert for a given new observation to record
        # its actions, but use the action from the learned model to move to the
        # next state / observation.
        learn_policy_fn = pytorch_policy_fn(args=args, model=model)
        expert_policy_fn = load_policy(filename=args.expert_policy_file)
        _, observations, _, expert_actions = run_model(
            args=args, policy_fn=learn_policy_fn,
            expert_policy_fn=expert_policy_fn)

        # 4. Aggregate the new data.
        train_dataset.add_data(observations=observations,
                               actions=expert_actions)
        kwargs = set_kwargs(args=args)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.min_batch_size,
                                  shuffle=True,
                                  **kwargs)


if __name__ == "__main__":
    args = get_args()
    run(args=args)
