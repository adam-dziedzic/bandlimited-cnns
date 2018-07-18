"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""
import sys
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.python import debug as tf_debug

from cnns.tf_tutorials.alexnet.alexnet import AlexNet

# switch backend to be able to save the graphic files on the servers
plt.switch_backend('agg')

seed = 37
tf.set_random_seed(37)

# read the input parameters
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--iterations", default=1, type=int,
                    help="number of iterations for the training")
parser.add_argument("-i", "--initbatchsize", default=32, type=int,
                    help="the initial size of the batch (number of "
                         "data points for a single forward and batch "
                         "passes")
parser.add_argument("-m", "--maxbatchsize", default=64, type=int,
                    help="the max size of the batch (number of data "
                         "points for a single forward and batch "
                         "passes")
parser.add_argument("-s", "--startsize", default=64, type=int,
                    help="the start size of the input")
parser.add_argument("-e", "--endsize", default=512, type=int,
                    help="the end size of the input")
parser.add_argument("-w", "--workers", default=0, type=int,
                    help="number of workers to fetch data for pytorch data loader, 0 means that the data will be "
                         "loaded in the main process")
parser.add_argument("-d", "--device", default="/device:GPU:0",
                    help="the type of device, e.g.: cpu, /cpu:0, /device:GPU:1, etc.")
parser.add_argument("-b", "--debug", default=False,
                    help="the type of device, e.g.: cpu, cuda:0, cuda:1, etc.")

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = '/path/to/train.txt'
val_file = '/path/to/val.txt'

# Learning params
learning_rate = 0.01
num_epochs = 1
batch_size = 32
data_size = 8192
input_size = 227
channel_size = 3

# Network params
dropout_rate = 0.5
num_classes = 10
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2',
                'conv1']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp/finetune_alexnet/tensorboard"
checkpoint_path = "tmp/finetune_alexnet/checkpoints"

is_validation = False
is_checkpoint = False

"""
Util functions.
"""


# Use standard TensorFlow operations to resize the image to a
# fixed shape.
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded,
                                           [input_size, input_size])
    return image_resized, label


"""
Main Part of the fine tuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

batch_sizes = []
times = []


def main(device):
    print("batch size: ", batch_size)
    batch_sizes.append(batch_size)
    print("reset default graph")
    tf.reset_default_graph()
    # Place data loading and preprocessing on the cpu
    with tf.device(device):
        # tr_data = ImageDataGenerator(train_file,
        #                              mode='training',
        #                              batch_size=batch_size,
        #                              num_classes=num_classes,
        #                              shuffle=True)
        # val_data = ImageDataGenerator(val_file,
        #                               mode='inference',
        #                               batch_size=batch_size,
        #                               num_classes=num_classes,
        #                               shuffle=False)

        # (x_train, y_train), (
        # x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # the input data: training data
        print("Generate the training data: ")
        data_x_train = tf.random_uniform(
            shape=[data_size, input_size, input_size, channel_size],
            seed=seed)
        # the input labels
        data_y_train = tf.random_uniform(shape=[data_size],
                                         minval=0,
                                         maxval=num_classes,
                                         dtype=tf.int32, seed=seed)
        # make the column as a one-hot vector
        # data_y_train = tf.feature_column.indicator_column(data_y_train)
        # convert label number into one-hot-encoding
        data_y_train = tf.one_hot(data_y_train, num_classes)
        # print("data_y_train shape: ", data_y_train)

        tr_data = tf.data.Dataset.from_tensor_slices(
            (data_x_train, data_y_train))
        tr_data = tr_data.map(_resize_function)

        # add batching to the data sources
        tr_data = tr_data.batch(batch_size)

        # create an reinitializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(
            tr_data.output_types,
            tr_data.output_shapes)
        next_batch = iterator.get_next()

        # Ops for initializing the two different iterators
        training_init_op = iterator.make_initializer(tr_data)

        if is_validation:
            # validation data
            print("Generate the validation data:")
            data_x_val = tf.random_uniform(
                [data_size, input_size, input_size, channel_size], seed=seed)
            data_y_val = tf.random_uniform(shape=[data_size], minval=0,
                                           maxval=num_classes,
                                           dtype=tf.int32, seed=seed)
            # data_y_val = tf.feature_column.indicator_column(data_y_val)
            # convert label number into one-hot-encoding
            data_y_val = tf.one_hot(data_y_val, num_classes)
            val_data = tf.data.Dataset.from_tensor_slices(
                (data_x_val, data_y_val))
            val_data = val_data.map(_resize_function)

            # add batching to the data sources
            val_data = val_data.batch(batch_size)

            validation_init_op = iterator.make_initializer(val_data)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32,
                       [batch_size, input_size, input_size, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    print("initialize the model")
    model = AlexNet(x, keep_prob, num_classes, train_layers,
                    input_size=input_size)

    # Link variable to model output
    score = model.fc8

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if
                v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                    labels=y))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable
        # variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    # train_batches_per_epoch = int(np.floor(data_size / batch_size))
    train_batches_per_epoch = 1
    val_batches_per_epoch = int(np.floor(data_size / batch_size))

    # Start Tensorflow session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        # model.load_initial_weights(sess)

        # To continue training from one of your checkpoints
        # saver.restore(sess, "/path/to/checkpoint/model.ckpt")

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)

            for step in range(train_batches_per_epoch):

                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)
                # print("label batch shape: ", len(label_batch))

                # sess.run({'conv1': model.conv1},
                #          feed_dict={x: img_batch,
                #                     y: label_batch,
                #                     keep_prob: dropout_rate})
                #
                #
                # sess.run({'conv5': model.conv5},
                #          feed_dict={x: img_batch,
                #                     y: label_batch,
                #                     keep_prob: dropout_rate})
                #
                # sess.run({'fc6': model.fc6},
                #          feed_dict={x: img_batch,
                #                     y: label_batch,
                #                     keep_prob: dropout_rate})

                # And run the training op

                start_total = time.time()
                sess.run({'fc8': score, 'train_op': train_op},
                         feed_dict={x: img_batch,
                                    y: label_batch,
                                    keep_prob: dropout_rate})
                # sess.run({'train_op': train_op})
                total_time = time.time() - start_total
                print("total running time: ", total_time)
                times.append(total_time)

                # Generate summary with the current batch of data and
                # write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})
                    writer.add_summary(s,
                                       epoch * train_batches_per_epoch + step)

            if is_validation:
                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                sess.run(validation_init_op)
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Validation Accuracy = {:.4f}".format(
                    datetime.now(), test_acc))

            if is_checkpoint:
                print("{} Saving checkpoint of model...".format(
                    datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(
                    checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(
                    datetime.now(), checkpoint_name))
    return batch_sizes, times


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    batch_size = args.initbatchsize
    device = args.device
    if tf.test.is_gpu_available() is False and (
            "GPU" in device or "gpu" in device):
        print("GPU is not available")
        device = "cpu:0"
    while batch_size <= args.maxbatchsize:
        batch_sizes, total_times = main(device)
        print("batch_sizes: ", batch_sizes)
        print("total_times: ", total_times)
        batch_size *= 2
