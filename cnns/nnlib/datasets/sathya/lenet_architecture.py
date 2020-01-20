import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, out=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, 25, 1)
        self.conv2 = nn.Conv1d(20, 50, 25, 1)
        self.fc1 = nn.Linear(5500, 500)
        self.fc2 = nn.Linear(500, out)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        # print('x size: ', x.size())
        x = x.view(-1, 5500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)