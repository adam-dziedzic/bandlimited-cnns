import tensorflow as tf
import numpy as np
from cnns.deeprl import tf_util
import time
from cnns.deeprl.utils import get_mean_std, read_data
from cnns.deeprl.models import create_model
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.utils.general_utils import PolicyType


def train_model(inputs, outputs, output_pred, input_ph, output_ph, env_name,
                sess, train_steps=10000):
    # create loss (mean squared error loss)
    # we compute the average loss over the mini-batch
    # the output_pred and output_ph have many data items - the size of the mini-batch
    # then we reduce over the mini-batch to get a single metric on how well our model
    # predicted the output
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # initialize variables
    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()

    # run training
    batch_size = 32
    sum_mse = 0
    count_mse = 0
    for training_step in range(train_steps):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
        input_batch = inputs[indices]
        output_batch = outputs[indices]

        # run the optimizer and get the mse
        # passing opt to the sess.run do the one time backprop
        _, mse_run = sess.run([opt, mse],
                              feed_dict={input_ph: input_batch,
                                         output_ph: output_batch})

        sum_mse += mse
        count_mse += 1

        # print the mse every so often
        if training_step % 1000 == 0:
            avg_mse = sum_mse / count_mse

            print(
                '{0:04d}; current mse; {1:.9f}; avg mse; {1:.9f}'.format(
                    training_step,
                    mse_run,
                    avg_mse))

            saver.save(sess, args.get_model_file())


def train_policy(args):
    obs, actions = read_data(
        args.expert_data_dir + "/" + args.env_name + '-' + str(
            args.rollouts) + ".pkl")
    mean, std = get_mean_std(obs)
    print("mean, std: ", mean, " ", std)
    input_size = obs.shape[1]
    output_size = actions.shape[1]
    with tf.Session() as sess:
        input_ph, output_ph, output_pred = create_model(
            mean=mean, std=std, input_size=input_size, output_size=output_size,
            hidden_units=args.hidden_units)
        train_model(inputs=obs, outputs=actions, input_ph=input_ph,
                    output_ph=output_ph, output_pred=output_pred,
                    env_name=args.env_name, sess=sess,
                    train_steps=args.train_steps)
    return input_size, output_size


def load_policy(args):
    obs, actions = read_data(
        args.expert_data_dir + "/" + args.env_name + ".pkl")
    input_size = obs.shape[1]
    output_size = actions.shape[1]
    mean, std = get_mean_std(obs)

    input_ph, output_ph, output_pred = create_model(
        mean=mean, std=std, input_size=input_size, output_size=output_size,
        hidden_units=args.hidden_units)

    # restore the saved model
    import os
    print('current working directory: ', os.getcwd())
    print('model file: ', args.get_model_file())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, args.get_model_file())

    policy_fn = tf_util.function([input_ph], output_pred)
    return policy_fn


if __name__ == "__main__":
    args = get_args()
    args.train_steps = 10000
    args.policy_type = PolicyType.PYTORCH_BEHAVE

    start = time.time()
    input_size, output_size = train_policy(args=args)
    stop = time.time()

    print('elapsed time (sec): ', stop - start)
    print("input size: ", input_size)
    print("output size: ", output_size)
