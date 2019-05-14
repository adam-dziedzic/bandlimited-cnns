from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging
import unittest
import numpy as np
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv2D
from cnns.tensorflow.conv2D_fft import Conv2D_fft
import keras.backend as K

import tensorflow as tf


def to_tf(input):
    """

    :param input: np array
    :return: channels last
    """
    return input.transpose(0, 2, 3, 1)


def from_tf(input):
    """

    :param input: np array
    :return: channels first
    """
    return input.transpose(0, 3, 1, 2)


class TestConv2D_fft(unittest.TestCase):

    def setUp(self):
        log_file = "conv2D_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        seed = 31
        np.random.seed(seed=seed)
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."
        self.strides = (1, 1)
        self.padding = 'valid'
        self.data_format = K.normalize_data_format(None)
        self.dilation_rate = (1, 1)
        self.dtype = np.float32

    def testSimple(self):
        tf.disable_eager_execution()
        inp = Input(shape=(1, 32, 32, 3))
        conv2D_fft = Conv2D_fft(3, 3)
        out = conv2D_fft(inp)
        model = Model(inp, out)
        output = model.predict(np.random.rand(1, 32, 32, 3))
        print("output shape: ", output.shape)
        self.logger.info(output.shape)

    def testEager(self):
        tf.enable_eager_execution()
        layer = Conv2D_fft(3, (2, 2))
        print(layer(tf.zeros([1, 32, 32, 3])))

    def testCompare(self):
        tf.enable_eager_execution()
        input = tf.random.uniform((1, 32, 32, 3))
        kernel = tf.random.uniform((1, 3, 3, 3))

        layer1 = Conv2D_fft(3, (3, 3))
        out1 = layer1.exec(x=input,
                           kernel=kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        result = out1.numpy()

        out2 = K.conv2d(x=input,
                        kernel=kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
        expect = out2.numpy()

        np.testing.assert_allclose(
            desired=expect, actual=result, rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test2(self):
        tf.enable_eager_execution()
        input = np.array([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                         dtype=self.dtype)
        kernel = np.array([[[[1.0, 2.0], [3.0, 2.0]]]],
                          dtype=self.dtype)  # F, C, HH, WW

        input = tf.convert_to_tensor(to_tf(input))
        # self.kernel_size + (input_dim, self.filters): HH, WW, C, F
        kernel = tf.convert_to_tensor(kernel.transpose(2, 3, 1, 0))

        layer1 = Conv2D_fft(1, (2, 2), use_bias=False)
        layer1.build_custom(input_shape=(1, 3, 3, 1), kernel=kernel)

        out1 = layer1.exec(x=input,
                           kernel=kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        result = out1.numpy()
        result = from_tf(result)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])

        np.testing.assert_allclose(
            desired=expect, actual=result, rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)


if __name__ == '__main__':
    unittest.main()
