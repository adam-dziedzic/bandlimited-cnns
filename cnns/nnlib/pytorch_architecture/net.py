import torch.nn as nn
import torch.nn.functional as F
from cnns.nnlib.pytorch_layers.conv_picker import Conv


def get_conv(args, in_channels, out_channels, kernel_size, stride=1,
             padding=0, bias=True):
    return Conv(kernel_sizes=[kernel_size], in_channels=in_channels,
                out_channels=[out_channels], strides=[stride],
                padding=[padding], args=args, is_bias=bias).get_conv()


H = 28
W = 28
num_classes = 10

kernel_size1 = 5
kernel_size2 = 5
in_channels = 1
out_channels1 = 20
hidden_neurons = 500
pull1 = 2
out_channels2 = 50
pull2 = 2

def get_HW_after_pull2(
        H=H,
        W=W,
        kernel_size1 = kernel_size1,
        kernel_size2 = kernel_size2,
        pull1= pull1,
        pull2=pull2):
    H_conv1 = H - (kernel_size1 - 1)
    W_conv1 = W - (kernel_size1 - 1)
    assert H_conv1 % 2 == 0
    assert W_conv1 % 2 == 0
    H_pull1 = H_conv1 // pull1  # 12
    W_pull1 = W_conv1 // pull1  # 12
    W_conv2 = W_pull1 - (kernel_size2 - 1)  # 8
    H_conv2 = H_pull1 - (kernel_size2 - 1)  # 8
    assert H_conv2 % pull2 == 0
    assert W_conv2 % pull2 == 0
    H_pull2 = H_conv2 // pull2  # 4
    W_pull2 = W_conv2 // pull2  # 4
    return H_pull2, W_pull2

conv1_param_nr = in_channels * out_channels1 * kernel_size1
conv2_param_nr = out_channels1 * out_channels2 * kernel_size2
conv_param_nr = conv1_param_nr + conv2_param_nr
# print('total conv params: ', conv_param_nr)

H_pull2, W_pull2 = get_HW_after_pull2()
fc1_param_nr = H_pull2 * W_pull2 * hidden_neurons
fc2_param_nr = hidden_neurons * num_classes
fc_param_nr = fc1_param_nr + fc2_param_nr
# print('total fully connected params: ', fc_param_nr)

param_nr = conv1_param_nr + conv2_param_nr + fc1_param_nr + fc2_param_nr
# print('total param nr: ', param_nr) # 18100

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1 = get_conv(args,
                              in_channels=in_channels,
                              out_channels=out_channels1,
                              kernel_size=kernel_size1,
                              stride=1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2 = get_conv(args,
                              in_channels=out_channels1,
                              out_channels=out_channels2,
                              kernel_size=kernel_size2,
                              stride=1)

        self.fc1 = nn.Linear(
            H_pull2 * W_pull2 * out_channels2,
            hidden_neurons)
        self.fc2 = nn.Linear(
            hidden_neurons,
            args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pull1, pull1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, pull2, pull2)
        x = x.view(-1, H_pull2 * W_pull2 * out_channels2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    print('Net for mnist dataset.')
