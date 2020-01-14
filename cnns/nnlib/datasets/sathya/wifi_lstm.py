import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from cnns.nnlib.datasets.sathya.torch_data import get_ucr
from cnns.nnlib.datasets.sathya.lenet_architecture import Net

HIDDEN_DIM = 512
delimiter = ';'


# Create the LSTM model
class LSTMWifi(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size):
        super(LSTMWifi, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes value vectors as inputs,
        # and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to output
        # the number of detected WiFis
        self.hidden2output = nn.Linear(hidden_dim, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        scores = self.hidden2output(lstm_out.view(len(input), -1))
        output = F.log_softmax(scores, dim=1)
        return output


def train(train_data, model):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    total = 0
    train_loss = 0
    correct = 0

    for iter, (inputs, targets) in enumerate(train_data):
        # print('iter: ', iter)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        # Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Run the forward pass to get predictions
        preds = model(inputs)

        # Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(preds, targets)
        train_loss += loss.item()
        total += targets.size(0)
        predicted = preds.argmax(1)
        correct += predicted.eq(targets).sum().item()

        loss.backward()
        optimizer.step()

    return correct / total, train_loss / total


def test(test_data, model):
    total = 0
    correct = 0

    for inputs, targets in test_data:

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        preds = model(inputs)
        predicted = preds.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct / total


def main(args, hidden_dim=HIDDEN_DIM):
    # model = LSTMWifi(input_dim=args.input_dim, hidden_dim=hidden_dim,
    #                  output_size=args.wifi)
    model = Net(out=args.wifi)

    if torch.cuda.is_available():
        model = model.to('cuda')

    train_data, test_data, _ = get_ucr(
        dataset_name=f'{args.wifi}_classes_WIFI',
        data_path=f'../sathya/data_journal/NLOS-6/',
        batch_size=args.batch_size,
        num_workers=0)

    header = ['epoch', 'test accuracy', 'train accuracy', 'train loss',
              'elapsed time']
    header_str = delimiter.join(header)
    print(header_str)
    for epoch in range(args.epochs):
        start_time = time.time()
        train_acc, train_loss = train(model=model, train_data=train_data)
        test_acc = test(model=model, test_data=test_data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        result = [epoch, test_acc, train_acc, train_loss, elapsed_time]
        result_str = delimiter.join([str(x) for x in result])
        print(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--wifi', metavar='N', type=int,
                        default=4,
                        help='number of wifis to detect')
    parser.add_argument('--epochs', metavar='N', type=int,
                        default=100,
                        help='number of epochs for training')
    parser.add_argument('--input_dim', type=int, default=512,
                        help='input dimension of the data')
    parser.add_argument('--batch_size', type=int,
                        default=256)
    parser.add_argument('--sample_count', type=int,
                        default=0)
    args = parser.parse_args()
    for wifi in [2, 3, 4, 5, 6]:
        args.wifi = wifi
        main(args)
