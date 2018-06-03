import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from cs231n.utils.perf_timing import *

print("Torch loaded")


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
print("cifar10_train: ", cifar10_train)

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                            transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)

dtype = torch.FloatTensor  # the CPU datatype

# Constant to control how frequently we print train loss
print_every = 10


# This is a little utility that we'll use to reset the model
# if we want to re-initialize all our parameters
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


"""
 Convolution returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride

H = 32, W = 32, pad = 0, HH = 7, WW = 7, stride = 2
H' = 1 + (32 + 2 * 0 - 7) / 2 = 13
W' = 1 + (32 + 2 * 0 - 7) / 2 = 13

"""
# Here's where we define the architecture of the model...
# after convolving 7x7 kernels with 32x32 pictures, we get 13x13 pictures
# we have 32 kernels (with depth 3), so the final total number of cells is: 13x13x32 = 5408
# after the first Conv2d layer we have the output with
simple_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
    nn.ReLU(inplace=True),
    Flatten(),  # see above for explanation, we get N pictures as vectors with 5408 feature values
    nn.Linear(5408, 10),  # affine layer - a fully connected layer
)
# nn.Linear: in_features – size of each input sample, out_features – size of each output sample

# Set the type of all data in this model to be FloatTensor
simple_model.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(simple_model.parameters(), lr=1e-2)  # lr sets the learning rate of the optimizer

fixed_model_base = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(5408, 1024),  # 5408=32*13*13 input size
    nn.ReLU(inplace=True),
    nn.Linear(1024, 10),
)

fixed_model = fixed_model_base.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.RMSprop(fixed_model.parameters(), lr=1e-2)

## Now we're going to feed a random batch into the model you defined and make sure the output is the right size
x = torch.randn(64, 3, 32, 32).type(dtype)
x_var = Variable(x.type(dtype))  # Construct a PyTorch Variable out of your input data
ans = fixed_model(x_var)  # Feed it through the model!

# Check to make sure what comes out of your model
# is the right dimensionality... this should be True
# if you've done everything correctly
is_equal = np.array_equal(np.array(ans.size()), np.array([64, 10]))
print("is equal dimensions (64,10): ", is_equal)

# Verify that CUDA is properly configured and you have a GPU available

print("is cuda available: ", torch.cuda.is_available())
gpu_dtype = torch.cuda.FloatTensor

if torch.cuda.is_available():
    import copy

    gpu_dtype = torch.cuda.FloatTensor

    fixed_model_gpu = copy.deepcopy(fixed_model_base).type(gpu_dtype)

    x_gpu = torch.randn(64, 3, 32, 32).type(gpu_dtype)
    x_var_gpu = Variable(x.type(gpu_dtype))  # Construct a PyTorch Variable out of your input data
    ans = fixed_model_gpu(x_var_gpu)  # Feed it through the model!

    # Check to make sure what comes out of your model
    # is the right dimensionality... this should be True
    # if you've done everything correctly
    np.array_equal(np.array(ans.size()), np.array([64, 10]))


    def test_time_gpu():
        torch.cuda.synchronize()  # Make sure there are no pending GPU computations
        ans = fixed_model_gpu(x_var_gpu)  # Feed it through the model!
        torch.cuda.synchronize()  # Make sure there are no pending GPU computations


    avg_time, _ = timeitrep(wrapper(fixed_model_gpu, x_var), number=5)
    print("average time gpu: ", avg_time)

avg_time, _ = timeitrep(wrapper(fixed_model, x_var), number=5)
print("average time cpu: ", avg_time)


def train(model, loss_fn, optimizer, num_epochs=1, dtype=torch.cuda.FloatTensor):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()  # set the model for the training mode
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)  # do the forward pass through the model

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0].item()))

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with respect to each
            # parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients computed by the backwards pass.
            optimizer.step()


def check_accuracy(model, loader, dtype=torch.cuda.FloatTensor):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


fixed_model_base = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(5408, 1024),  # 5408=32*13*13 input size
    nn.ReLU(inplace=True),
    nn.Linear(1024, 10),
)

dtype = torch.FloatTensor  # torch.cuda.FloatTensor
cpu_model = fixed_model_base.type(dtype)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(cpu_model.parameters(), lr=1e-2)

total_epochs = 10
for epoch in range(total_epochs):
    train(cpu_model, loss_fn, optimizer, num_epochs=1, dtype=dtype)
    check_accuracy(cpu_model, loader_val, dtype=dtype)
