from builtins import object

from cs231n.layer_utils import *


class ThreeLayerConvNetFFT1D(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, W)
    consisting of N images, each transformed to an array of W values with C input
    channels.
    """

    def __init__(self, input_dim=(3, 1024), num_filters=32, filter_size=49,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, filter_channels=3, pad_convolution=None,
                 stride_convolution=1, pool_stride=2, pool_width=2, energy_rate_convolution=1.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        C, W = input_dim
        self.stride_convolution = stride_convolution
        if pad_convolution is None:
            self.pad_convolution = (filter_size - 1) // 2
        else:
            self.pad_convolution = pad_convolution
        self.pool_stride = pool_stride
        self.pool_width = pool_width
        self.energy_rate_convolution = energy_rate_convolution

        self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, filter_channels, filter_size])
        self.params['b1'] = np.zeros([num_filters])
        dim_width_conv = (1 + (W + 2 * self.pad_convolution - filter_size) // self.stride_convolution)
        dim_width_pool = np.int(((dim_width_conv - pool_width) // pool_stride) + 1)

        self.params['W2'] = np.random.normal(0, weight_scale,
                                             [dim_width_pool * num_filters, hidden_dim])
        # self.params['W2'] = np.random.normal(0, weight_scale, [np.int(H/2)*np.int(W/2)*num_filters, hidden_dim])
        # print("shape of W2: ", self.params['W2'].shape)
        self.params['b2'] = np.zeros([hidden_dim])
        self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b3'] = np.zeros([num_classes])

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        conv_param = {'stride': self.stride_convolution, 'pad': self.pad_convolution,
                      'preserve_energy_rate': self.energy_rate_convolution}
        pool_param = {'pool_width': self.pool_width, 'stride': self.pool_stride}


        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        maxpool1_out, combined_cache = conv_relu_pool_forward_fft_1D(X, W1, b1, conv_param, pool_param)

        affine1_out, affine1_cache = affine_forward(maxpool1_out, W2, b2)
        relu2_out, relu2_cache = relu_forward(affine1_out)

        affine2_out, affine2_cache = affine_forward(relu2_out, W3, b3)

        scores = np.copy(affine2_out)

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dsoft = softmax_loss(scores, y)

        loss += self.reg * 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        dx3, dw3, db3 = affine_backward(dsoft, affine2_cache)
        drelu2 = relu_backward(dx3, relu2_cache)
        dx2, dw2, db2 = affine_backward(drelu2, affine1_cache)
        dx1, dw1, db1 = conv_relu_pool_backward_fft_1D(dx2, combined_cache)

        grads['W3'], grads['b3'] = dw3 + self.reg * W3, db3
        grads['W2'], grads['b2'] = dw2 + self.reg * W2, db2
        grads['W1'], grads['b1'] = dw1 + self.reg * W1, db1

        return loss, grads
