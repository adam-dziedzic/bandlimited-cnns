import numpy as np
import tensorflow as tf


class spectral_conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel,
                 kernel_size, random_seed, data_format='NHWC', index=0):
        """
        A convolutional layer with spectrally-parameterized weights.

        :param input_x: Should be a 4D array like:
                            (batch_num, channel_num, img_len, img_len)
        :param in_channel: The number of channels
        :param out_channel: number of filters required
        :param kernel_size: kernel size
        :param random_seed: random seed
        :param data_format: image should be with CHANNEL LAST: NHWC
        :param index: The layer index used for naming
        """
        assert len(input_x.shape) == 4
        if data_format == 'NHWC':
            assert input_x.shape[1] == input_x.shape[2]
            assert input_x.shape[3] == in_channel
        elif data_format == 'NCHW':
            assert input_x.shape[1] == in_channel
            assert input_x.shape[2] == input_x.shape[3]

        def _glorot_sample(kernel_size, n_in, n_out):
            limit = np.sqrt(6 / (n_in + n_out))
            return np.random.uniform(
                low=-limit,
                high=limit,
                size=(n_in, n_out, kernel_size, kernel_size)
            )

        with tf.variable_scope('spec_conv_layer_{0}'.format(index)):
            with tf.name_scope('spec_conv_kernel'):
                samp = _glorot_sample(kernel_size, in_channel, out_channel)
                """
                tf.fft2d: Computes the 2-dimensional discrete Fourier transform 
                over the inner-most 2 dimensions of input.
                """
                # shape channel_in, channel_out, kernel_size, kernel_size
                spectral_weight_init = tf.fft2d(samp)

                real_init = tf.get_variable(
                    name='real_{0}'.format(index),
                    initializer=tf.real(spectral_weight_init))

                imag_init = tf.get_variable(
                    name='imag_{0}'.format(index),
                    initializer=tf.imag(spectral_weight_init))

                spectral_weight = tf.complex(
                    real_init,
                    imag_init,
                    name='spectral_weight_{0}'.format(index)
                )
                self.spectral_weight = spectral_weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(
                    name='conv_bias_{0}'.format(index),
                    shape=b_shape,
                    initializer=tf.glorot_uniform_initializer(
                        seed=random_seed
                    ))
                self.bias = bias

            """
            ifft2d: Computes the inverse 2-dimensional discrete Fourier 
            transform over the inner-most 2 dimensions of input.
            """
            complex_spatial_weight = tf.ifft2d(spectral_weight)
            spatial_weight = tf.real(
                complex_spatial_weight,
                name='spatial_weight_{0}'.format(index)
            )

            # we need kernel tensor of shape [filter_height, filter_width,
            # in_channels, out_channels]
            self.weight = tf.transpose(spatial_weight, [2, 3, 0, 1])

            conv_out = tf.nn.conv2d(input_x, spatial_weight,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    data_format=data_format)
            self.cell_out = tf.nn.relu(
                tf.nn.bias_add(conv_out, bias, data_format=data_format))

    def output(self):
        return self.cell_out
