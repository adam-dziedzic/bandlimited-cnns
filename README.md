## Machine learning and databases for time-series

Use machine learning and database technologies to process the time-series data.

# Important files:

## Memory management:
`cnns/pytorch_tutorials/memory_net.py`

Find cliffs in the execution of neural networks.
Go to the level of C++ and cuda.
Find for what input size, the memory size is not sufficient.
Run a single forward pass and a subsequent backward pass.

Define neural network, compute loss and make updates to the weights of the
network.


We do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. ExperimentSpectralSpatial the network on the test data