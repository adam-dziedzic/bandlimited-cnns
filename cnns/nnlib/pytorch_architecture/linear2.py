import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear2(nn.Module):
    def __init__(self, args, hidden_sizes=[512, 128, 64, 16, 8]):
        super(Linear2, self).__init__()
        self.args = args
        self.input_size = args.input_size
        # num_classes: number of output classes.
        self.num_classes = args.num_classes
        # in_channels: number of channels in the input data.
        self.in_channels = args.in_channels
        self.dtype = args.dtype

        self.fc1 = nn.Linear(self.input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], self.num_classes)

    def forward(self, x):
        x = torch.squeeze(x)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        x = F.sigmoid(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)
