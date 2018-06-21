import tensorflow as tf
import numpy as np
import math
import time

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):

    # Set up variables
    Wconv1 = tf.get_variable("Wconv1",shape=[7,7,3,32])
    bconv1 = tf.get_variable("bconv1",shape=[32])
    Waffine1 = tf.get_variable("Waffine1", shape=[5408,1024])
    baffine1 = tf.get_variable("baffine1", shape=[1024])
    Waffine2 = tf.get_variable("Waffine2", shape=[1024,10])
    baffine2 = tf.get_variable("baffine2", shape=[10])

    # Define model
    conv1act = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    relu1act = tf.nn.relu(conv1act)
    batchnorm1act = tf.layers.batch_normalization(relu1act, training=is_training)  # Using the tf.layers batch norm as its easier
    pool1act = tf.nn.max_pool(batchnorm1act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flatten1 = tf.reshape(pool1act,[-1,5408])
    affine1act = tf.matmul(flatten1, Waffine1) + baffine1
    relu2act = tf.nn.relu(affine1act)
    y_out = tf.matmul(relu2act, Waffine2) + baffine2

    return y_out
    
y_out = complex_model(X,y,is_training)

x = np.random.randn(64, 32, 32, 3)

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        tf.global_variables_initializer().run()
        start = time.time()
        ans = sess.run(y_out, feed_dict={X:x, is_training:True})
        print("running time: ", time.time() - start)

