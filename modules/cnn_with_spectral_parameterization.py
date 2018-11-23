import os
import socket
import time
from enum import Enum

import numpy as np
import tensorflow as tf

from .image_generator import ImageGenerator
from .layers import spectral_conv_layer

################################################################################
# Convenience functions and classes (for residual connections and Enums.       #
################################################################################
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, is_training=True, data_format="channels_last"):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)


class EnumWithNames(Enum):
    """
    The Enum classes that inherit from the EnumWithNames will get the get_names
    method to return an array of strings representing all possible enum values.
    """

    @classmethod
    def get_names(cls):
        return [enum_value.name for enum_value in cls]


class OptimizerType(EnumWithNames):
    MOMENTUM = 1
    ADAM = 2


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Arguments:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Arguments:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.

    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


class CNN_Spectral_Param():

    def __init__(self,
                 architecture,
                 num_output=10,
                 use_spectral_params=True,
                 kernel_size=3,
                 l2_norm=0.01,
                 learning_rate=1e-4,
                 data_format='NHWC',
                 random_seed=0,
                 is_residual=True,
                 is_training=True):

        """
        :param architecture: Defines which architecture to build (either deep or
        generic)
        :param num_output: Number of classes to predict
        :param use_spectral_params: Flag to turn spectral parameterization on
        and off
        :param kernel_size: size of convolutional kernel
        :param l2_norm: Scale factor for l2 norm of CNN weights when calculating
        l2 loss
        :param learning_rate: Learning rate for Adam AdamOptimizer
        :param data_format: Format of input images, either 'NHWC' or 'NCHW'
        :param random_seed: random seed for parameter initialization
        :random_seed: Seed for initializers to create reproducable results
        :param is_residual: should the residual connections be added to the
        network
        :param is_training: a Boolean for whether the model is in training or
        inference mode. Needed for batch normalization.
        """
        self.num_output = num_output
        self.architecture = architecture
        self.use_spectral_params = use_spectral_params
        self.kernel_size = kernel_size
        self.l2_norm = l2_norm
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.data_format = data_format
        self.is_residual = is_residual
        self.is_training = is_training

        # Variables to track different metrics we're interested in
        self.loss = []
        self.train_err = []

    """
    This class builds and trains the generic and deep CNN architectures
    as described in section 5.2 of the paper with and without spectral pooling.
    """

    def build_graph(self, input_x, input_y):
        """
        This function calls one of two helper functions to build the CNN graph

        :param input_x: 4D array containing images to train model on
        :param input_y: 1D array containing class labels of images
        """
        if self.architecture == 'generic':
            return self._build_generic_architecture(input_x, input_y)
        elif self.architecture == 'deep':
            return self._build_deep_architecture(input_x, input_y)
        elif self.architecture == 'deep_residual1':
            return self._build_deep_residual_architecture1(input_x, input_y)
        elif self.architecture == 'deep_residual2':
            return self._build_deep_residual_architecture2(input_x, input_y)
        elif self.architecture == 'deep_residual3':
            return self._build_deep_residual_architecture3(input_x, input_y)
        elif self.architecture == 'deep2':
            return self._build_deep2(input_x, input_y)
        elif self.architecture == 'deep3':
            return self._build_deep3(input_x, input_y)
        elif self.architecture == 'deep2_bn2':
            return self._build_deep2_bn2(input_x, input_y)
        elif self.architecture == 'deep_bn':
            return self._build_deep_bn_architecture(input_x, input_y)
        else:
            raise ValueError(
                'Architecture \'' + self.architecture + '\' not defined')

    def train_step(self, loss, optimizer_type, batch_size, num_images,
                   num_epochs):
        """
        Calls a selected optimizer to minimize the loss.

        :param loss: the loss to minimize
        :param num_images: number of images for training
        """
        with tf.name_scope('train_step'):
            if optimizer_type is OptimizerType.ADAM:
                step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            elif optimizer_type is OptimizerType.MOMENTUM:
                if self.use_spectral_params:
                    batch_denom = 12800
                else:
                    batch_denom = 128

                learning_rate_fn = learning_rate_with_decay(
                    batch_size=batch_size,
                    batch_denom=batch_denom,
                    num_images=num_images,
                    boundary_epochs=[
                        int(num_epochs * 0.65),
                        int(num_epochs * 0.95)],
                    decay_rates=[1, 0.1,
                                 0.01])

                """global_step refers to the number of batches seen by the graph. Every time
                a batch is provided, the weights are updated in the direction that
                minimizes the loss. global_step just keeps track of the number of batches
                seen so far: https://bit.ly/2AAqjs1"""
                global_step = tf.train.get_or_create_global_step()
                learning_rate = learning_rate_fn(global_step)
                step = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                  momentum=0.9,
                                                  use_nesterov=True).minimize(
                    loss)
            else:
                raise ValueError("Unknown optimizer type")

        return step

    def evaluate(self, pred, input_y):
        """
        Calculates the number of errors made in the prediction array and the
        accuracy rate.
        :param pred: The prediction array for the class labels
        :param input_y: The ground-truth y values
        :return: error_num, error_rate
        """
        with tf.name_scope('evaluate'):
            error_num = tf.count_nonzero(pred - input_y, name='error_num')
            tf.summary.scalar(self.architecture + '_error_num', error_num)
        return error_num

    def train(self, X_train, y_train, batch_size, num_epochs, optimizer_type,
              num_images, restore_checkpoint=None,
              model_name='spectral_params'):
        """
        Trains the CNN model. This is where data augmentation is added and
        the training accuracy is tracked.

        :param X_train: 4D training set (num images, height, width, num channels)
        :param y_train: 1D training labels
        :param batch_size: Number of images to include in the minibatch,
        before applying gradient updates
        :param num_epochs: Number of epochs to train the model for
        :param optimizer_type: the type of optimizer, e.g., Adam, Momentum
        :param num_images: the number of images for training
        """
        full_model_name = '{0}_{1}'.format(model_name, time.time())

        self.full_model_name = full_model_name

        # Instantiate image generator for data augmentation
        img_gen = ImageGenerator(X_train, y_train)
        img_gen.translate(shift_height=-2, shift_width=0)
        generator = img_gen.next_batch_gen(batch_size)

        results_dir = "results"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

            output, loss = self.build_graph(xs, ys)

            iters = int(X_train.shape[0] / batch_size)
            # print('number of batches for training: {}'.format(iters))
            start_time = time.time()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                step = self.train_step(loss, batch_size=batch_size,
                                       optimizer_type=optimizer_type,
                                       num_epochs=num_epochs,
                                       num_images=num_images)
            pred = tf.argmax(output, axis=1)
            eve = self.evaluate(pred, ys)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                merge = tf.summary.merge_all()
                writer = tf.summary.FileWriter("log/{}/{}".format(
                    model_name,
                    full_model_name
                ), sess.graph)
                saver = tf.train.Saver()

                sess.run(init)

                date_str = time.asctime().replace(" ", "-").replace(":", "-")
                out_file = "results/running_metrics_spectral_param_" + \
                           self.architecture + "_" + date_str
                info = "batch_size," + str(batch_size) + ",num_epochs," + str(
                    num_epochs) + ",optimizer_type," + str(
                    optimizer_type) + ",num_images," + str(
                    num_images) + ",architecture," + str(
                    self.architecture) + ",is spectral," + str(
                    self.use_spectral_params) + ",is residual," + str(
                    self.is_residual) + ",machine name," + str(
                    socket.gethostname()) + "\n"
                print(info)
                with open(out_file, 'a') as f:
                    f.write(info)

                iter_total = 0
                for epc in range(num_epochs):
                    # Apply vertical translations and random horizontal flips
                    if epc % 4 == 0 or epc % 4 == 1:
                        img_gen.translate(shift_height=2, shift_width=0)
                    elif epc % 4 == 2 or epc % 4 == 3:
                        img_gen.translate(shift_height=-2, shift_width=0)

                    if np.random.randint(2, size=1)[0] == 1:
                        img_gen.flip(mode='h')

                    loss_in_epoch = []
                    error_rate_in_epoch = []
                    num_data_points = 0
                    for itr in range(iters):
                        iter_total += 1

                        training_batch_x, training_batch_y = next(generator)
                        num_data_points += len(training_batch_y)

                        _, cur_loss, error_num = sess.run(
                            [step, loss, eve],
                            feed_dict={xs: training_batch_x,
                                       ys: training_batch_y})
                        loss_in_epoch.append(cur_loss)
                        error_rate_in_epoch.append(error_num)

                    self.loss.append(np.mean(loss_in_epoch))
                    self.train_err.append(
                        np.sum(error_rate_in_epoch) / num_data_points)

                    result = ['epoch', epc,
                              'Train error', self.train_err[-1],
                              'Loss', self.loss[-1],
                              'Elapsed time', time.time() - start_time]
                    results_str = ",".join([str(x) for x in result])
                    print(results_str)
                    with open(out_file, 'a') as f:
                        f.write(results_str + "\n")

    def _build_generic_architecture(self, input_x, input_y):
        """
        Builds the generic architecture (defined in section 5.2 of the paper)

        This architecture is a pair of convolution and max-pool layers, followed
        by three fully-connected layers and a softmax.

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # These if statements decide whether we'll use spectral convolution or
        # the built-in tensorflow convolutional layer
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)

        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)

        else:
            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv2')

        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        flatten = tf.contrib.layers.flatten(inputs=pool2)

        fc1 = tf.contrib.layers.fully_connected(inputs=flatten,
                                                num_outputs=1024,
                                                activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=512,
                                                activation_fn=tf.nn.relu)
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                num_outputs=self.num_output,
                                                activation_fn=None)

        fc_weights = [v for v in tf.trainable_variables() if
                      'weights' in v.name]

        with tf.name_scope("loss"):
            # Calculating l2 norms for the loss
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels])

            l2_loss += tf.reduce_sum([tf.norm(w) for w in fc_weights])

            # Calculating cross entropy loss
            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=fc3),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        # Returns output of the final layer as well as the loss
        return fc3, loss

    def _build_deep_architecture(self, input_x, input_y):
        """
        Builds the deep architecture (defined in section 5.2 of the paper)

        This architecture is defined as follows:
            back-to-back convolutions, max-pool, back-to-back-to-back
            convolutions,
            max-pool, back-to-back 1-filter convolutions,
            and a global averaging, softmax

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv2')

        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv5')

        pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=6)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=None,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def projection_shortcut(self, inputs, filters_out, strides=1):
        # The padding is consistent and is based only on `kernel_size`, not on
        # the dimensions of `inputs` (as opposed to using `tf.layers.conv2d`
        # alone).
        kernel_size = 1
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size=kernel_size,
                                   data_format=self.data_format)

        if self.data_format == "NCHW":
            data_format = "channels_first"
        else:
            data_format = "channels_last"

        # tf.logging.info("conv_type: " + self.conv_type.name)
        return tf.layers.conv2d(
            inputs=inputs, filters=filters_out, kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def _build_deep_residual_architecture1(self, input_x, input_y):
        """
        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv2')

        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        # add the second residual connection
        shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=6)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=None,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep_residual_architecture2(self, input_x, input_y):
        """
        Set RELU after residual connection.

        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1,
                                           is_relu=True)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2,
                                           is_relu=False)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv2')

        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        # add RELU after the residual connection
        conv2 = tf.nn.relu(conv2)

        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        # add the second residual connection
        shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3,
                                           is_relu=True)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4,
                                           is_relu=True)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5,
                                           is_relu=False)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        conv5 = tf.nn.relu(conv5)

        pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=6,
                                           is_relu=True)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7,
                                           is_relu=True)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep_residual_architecture3(self, input_x, input_y):
        """
        Pure resnet without any projections.
        Set RELU after residual connection.

        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        # shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1,
                                           is_relu=True)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        shortcut1 = conv1

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2,
                                           is_relu=False)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv2')

        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        # add RELU after the residual connection
        conv2 = tf.nn.relu(conv2)

        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        # add the second residual connection
        # shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3,
                                           is_relu=True)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        shortcut2 = conv3

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4,
                                           is_relu=True)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5,
                                           is_relu=False)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        conv5 = tf.nn.relu(conv5)

        pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=6,
                                           is_relu=True)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7,
                                           is_relu=True)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep2(self, input_x, input_y):
        """
        Remove Max Pooling and replace it with strides in boundary convolutions.
        For final layer add avg pooling with fully connected.

        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        # shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1,
                                           is_relu=True)
            conv0 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv0 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv0')

        shortcut1 = conv0

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv0,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2,
                                           is_relu=True)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=conv0,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3,
                                           is_relu=False)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv2')
        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        # add RELU after the residual connection
        conv2 = tf.nn.relu(conv2)

        # conv2 = tf.layers.max_pooling2d(inputs=conv2,
        #                                 pool_size=3,
        #                                 strides=2,
        #                                 padding='SAME',
        #                                 name='max_pool_1')

        # add the second residual connection
        # shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv2,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4,
                                           is_relu=True,
                                           strides=(1, 2, 2, 1))
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3',
                                     strides=(2, 2))

        shortcut2 = conv3

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5,
                                           is_relu=True)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=6,
                                           is_relu=False)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        conv5 = tf.nn.relu(conv5)

        # conv5 = tf.layers.max_pooling2d(inputs=conv5,
        #                                 pool_size=3,
        #                                 strides=2,
        #                                 padding='SAME',
        #                                 name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv5,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7,
                                           is_relu=True,
                                           strides=(1, 2, 2, 1))
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=conv5,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6',
                                     strides=(2, 2))

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=8,
                                           is_relu=True)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep3(self, input_x, input_y):
        """
        Add one more convolution in the first layer of convolutions (before the
        first Max Pool layer). Very similar to the deep model, only better
        adjusted to the residual connections.

        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        # shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1,
                                           is_relu=True)
            conv0 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv0 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv0')

        shortcut1 = conv0

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv0,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2,
                                           is_relu=True)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=conv0,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3,
                                           is_relu=False)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv2')
        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        # add RELU after the residual connection
        conv2 = tf.nn.relu(conv2)

        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        # add the second residual connection
        # shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4,
                                           is_relu=True)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        shortcut2 = conv3

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5,
                                           is_relu=True)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=conv3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=6,
                                           is_relu=False)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=conv4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        conv5 = tf.nn.relu(conv5)

        pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7,
                                           is_relu=True)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=8,
                                           is_relu=True)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=conv6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep2_bn2(self, input_x, input_y):
        """
        Based on deep2 - add the batch normalization and RELU before
        the convolutions in the residual building block. This is similar to
        building block v2 in ResNet.

        Remove Max Pooling and replace it with strides in boundary convolutions.
        For final layer add avg pooling with fully connected.

        Builds the deep residual architecture (defined in section 5.2 of the
        paper and augmented with residual connections).

        This architecture is defined as follows:
            back-to-back convolutions,
            max-pool,
            back-to-back-to-back
            convolutions,
            max-pool,
            back-to-back 1-filter convolutions,
            and a global averaging

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # this is for the first residual connection
        # shortcut1 = self.projection_shortcut(input_x, filters_out=96)

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1,
                                           is_relu=False)
            conv0 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv0 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv0')

        shortcut1 = conv0

        inputs = conv0
        inputs = batch_norm(inputs, self.is_training, self.data_format)
        inputs = tf.nn.relu(inputs)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=inputs,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2,
                                           is_relu=False)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=inputs,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv1')

        inputs2 = conv1
        inputs2 = batch_norm(inputs2, self.is_training, self.data_format)
        inputs2 = tf.nn.relu(inputs2)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=inputs2,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3,
                                           is_relu=False)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=inputs2,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv2')
        if self.is_residual:
            # add the residual connection
            conv2 += shortcut1

        # add RELU after the residual connection
        conv2 = tf.nn.relu(conv2)

        # conv2 = tf.layers.max_pooling2d(inputs=conv2,
        #                                 pool_size=3,
        #                                 strides=2,
        #                                 padding='SAME',
        #                                 name='max_pool_1')

        # add the second residual connection
        # shortcut2 = self.projection_shortcut(pool1, filters_out=192)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv2,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4,
                                           is_relu=False,
                                           strides=(1, 2, 2, 1))
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv3',
                                     strides=(2, 2))

        shortcut2 = conv3

        inputs3 = conv3
        inputs3 = batch_norm(inputs3, self.is_training, self.data_format)
        inputs3 = tf.nn.relu(inputs3)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=inputs3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5,
                                           is_relu=False)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=inputs3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv4')

        inputs4 = conv4
        inputs4 = batch_norm(inputs4, self.is_training, self.data_format)
        inputs4 = tf.nn.relu(inputs4)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=inputs4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=6,
                                           is_relu=False)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=inputs4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=None,
                                     padding='SAME',
                                     name='conv5')

        if self.is_residual:
            conv5 += shortcut2  # the 2nd residual connection

        conv5 = tf.nn.relu(conv5)

        # conv5 = tf.layers.max_pooling2d(inputs=conv5,
        #                                 pool_size=3,
        #                                 strides=2,
        #                                 padding='SAME',
        #                                 name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=conv5,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7,
                                           is_relu=False,
                                           strides=(1, 2, 2, 1))
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=conv5,
                                     filters=192,
                                     kernel_size=1,
                                     activation=None,
                                     padding='SAME',
                                     name='conv6',
                                     strides=(2, 2))

        inputs6 = conv6
        inputs6 = batch_norm(inputs6, self.is_training, self.data_format)
        inputs6 = tf.nn.relu(inputs6)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=inputs6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=8,
                                           is_relu=True)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=inputs6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        self.global_avg = tf.reduce_mean(input_tensor=conv7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss

    def _build_deep_bn_architecture(self, input_x, input_y):
        """
        Builds the deep architecture (defined in section 5.2 of the paper)
        with batch normalization (after convolution and its RELU).

        This is a bit unusual - we added batch normalization after RELU.

        This architecture is defined as follows:
            back-to-back convolutions, max-pool, back-to-back-to-back
            convolutions,
            max-pool, back-to-back 1-filter convolutions,
            and a global averaging, softmax

        :param input_x: 4D training set
        :param input_y: 1D training labels
        """
        spatial_conv_weights = []

        # Again, these if-statements determine whether to use spectral conv or
        # default
        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=input_x,
                                           in_channel=3,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=1)
            conv1 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv1 = tf.layers.conv2d(inputs=input_x,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv1')

        batch1 = batch_norm(conv1, self.is_training, self.data_format)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=batch1,
                                           in_channel=96,
                                           out_channel=96,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=2)
            conv2 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv2 = tf.layers.conv2d(inputs=batch1,
                                     filters=96,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv2')

        batch2 = batch_norm(conv2, self.is_training, self.data_format)

        pool1 = tf.layers.max_pooling2d(inputs=batch2,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_1')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool1,
                                           in_channel=96,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=3)
            conv3 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv3 = tf.layers.conv2d(inputs=pool1,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv3')

        batch3 = batch_norm(conv3, self.is_training, self.data_format)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=batch3,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=4)
            conv4 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv4 = tf.layers.conv2d(inputs=batch3,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv4')

        batch4 = batch_norm(conv4, self.is_training, self.data_format)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=batch4,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=self.kernel_size,
                                           random_seed=self.random_seed,
                                           index=5)
            conv5 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv5 = tf.layers.conv2d(inputs=batch4,
                                     filters=192,
                                     kernel_size=self.kernel_size,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv5')

        batch5 = batch_norm(conv5, self.is_training, self.data_format)

        pool2 = tf.layers.max_pooling2d(inputs=batch5,
                                        pool_size=3,
                                        strides=2,
                                        padding='SAME',
                                        name='max_pool_2')

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=pool2,
                                           in_channel=192,
                                           out_channel=192,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=6)
            conv6 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv6 = tf.layers.conv2d(inputs=pool2,
                                     filters=192,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv6')

        batch6 = batch_norm(conv6, self.is_training, self.data_format)

        if self.use_spectral_params:
            sc_layer = spectral_conv_layer(input_x=batch6,
                                           in_channel=192,
                                           out_channel=10,
                                           kernel_size=1,
                                           random_seed=self.random_seed,
                                           index=7)
            conv7 = sc_layer.output()
            spatial_conv_weights.append(sc_layer.weight)
        else:
            conv7 = tf.layers.conv2d(inputs=batch6,
                                     filters=10,
                                     kernel_size=1,
                                     activation=tf.nn.relu,
                                     padding='SAME',
                                     name='conv7')

        batch7 = batch_norm(conv7, self.is_training, self.data_format)

        self.global_avg = tf.reduce_mean(input_tensor=batch7, axis=[1, 2])

        with tf.name_scope("loss"):
            if self.use_spectral_params:
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in spatial_conv_weights if
                     w.shape[0] == 1])
            else:
                conv_kernels = [v for v in tf.trainable_variables() if
                                'kernel' in v.name]
                l2_loss = tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 3])
                l2_loss += tf.reduce_sum(
                    [tf.norm(w, axis=[-2, -1]) for w in conv_kernels if
                     w.shape[0] == 1])

            label = tf.one_hot(input_y, self.num_output)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                        logits=self.global_avg),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, self.l2_norm * l2_loss,
                          name='loss')

        return self.global_avg, loss
