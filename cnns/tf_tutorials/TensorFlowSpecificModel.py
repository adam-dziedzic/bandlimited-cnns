"""
Training a specific model

In this section, we are going to specify a model for you to construct. The goal here is not to get good performance (that will be next), but instead to get comfortable with understanding the TensorFlow documentation and configuring your own model.

Using the code provided above as guidance, and using the following TensorFlow documentation, specify a model with the following architecture:

7x7 Convolutional Layer with 32 filters and stride of 1
ReLU Activation Layer
Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
2x2 Max Pooling layer with a stride of 2
Affine layer with 1024 output units
ReLU Activation Layer
Affine layer from 1024 input units to 10 outputs
"""
import time
from TensorFlowStanford import *

# clear oldConvNetFFT1D variables
# tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
# X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# y = tf.placeholder(tf.int64, [None])

# # define model
# def complex_model(X,y,is_training):
    
#     # Set up variables
#     Wconv1 = tf.get_variable("Wconv1",shape=[7,7,3,32])
#     bconv1 = tf.get_variable("bconv1",shape=[32])
#     Waffine1 = tf.get_variable("Waffine1", shape=[5408,1024])
#     baffine1 = tf.get_variable("baffine1", shape=[1024])
#     Waffine2 = tf.get_variable("Waffine2", shape=[1024,10])
#     baffine2 = tf.get_variable("baffine2", shape=[10])
    
#     # Define model
#     conv1act = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
#     relu1act = tf.nn.relu(conv1act)
#     batchnorm1act = tf.layers.batch_normalization(relu1act, training=is_training)  # Using the tf.layers batch norm as its easier
#     pool1act = tf.nn.max_pool(batchnorm1act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
#     flatten1 = tf.reshape(pool1act,[-1,5408])
#     affine1act = tf.matmul(flatten1, Waffine1) + baffine1
#     relu2act = tf.nn.relu(affine1act)
#     y_out = tf.matmul(relu2act, Waffine2) + baffine2
    
#     return y_out

def run_complex_model():
    # Now we're going to feed a random batch into the model 
    # and make sure the output is the right size
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)
    x = np.random.randn(64, 32, 32, 3)
    with tf.Session() as sess:
        with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()

            ans = sess.run(y_out,feed_dict={X:x,is_training:True})
            start = time.time()
            sess.run(y_out,feed_dict={X:x,is_training:True})
            total = time.time() - start
            print("total execution of the session run: ", total, " sec")
            print(ans.shape)
            print(np.array_equal(ans.shape, np.array([64, 10])))


if __name__ == "__main__":
    run_complex_model()
