import os
import pickle
import tensorflow as tf
import numpy as np
from cnns.deeprl import tf_util
from cnns.nnlib.utils.general_utils import PolicyType


def create_model(mean, std, input_size=1, output_size=1, hidden_units=100):
    # create input PlaceHolder
    # in the behavioral clonning case, you would use, for example, 7 instead of 1 for the
    # last dimension, since 7 represents 7 joins of our robot
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_size])

    # normalize the input
    input_ph_norm = (input_ph - mean) / (std + 1e-6)

    # create the output PlaceHolder: this is what we expect as the output and what we use
    # to compute the error and back-propagate it.
    # for the output in the behavioral cloning algorithm you could use 4 for the 4 possible
    # actions to be takes for each of the 2 joins, apply the force to the right or left, we
    # would use 4 instead of 1 in the second dimension in the output_ph
    # the ground truth labels are fed into the output_ph placeholdergT
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

    # create variables
    W0 = tf.get_variable(
        name='W0', shape=[input_size, hidden_units],
        initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(
        name='W1', shape=[hidden_units, hidden_units],
        initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(
        name='W2', shape=[hidden_units, output_size],
        initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[hidden_units],
                         initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[hidden_units],
                         initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[1],
                         initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    """
    We want to predict the values of the sin function, which has the output values in the
    range from -1 to +1, so we do not use relu non-linearit in the last layer because it would
    limit our final output to only the positive values. However, we could imagine using the 
    tanh non-linearity that has the scope from -1 to + 1. 
    """
    # activations = [tf.nn.relu, tf.nn.relu, None]
    # activations = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    # activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
    activations = [tf.nn.tanh, tf.nn.tanh, None]

    # create computation graph
    layer = input_ph_norm
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred


def run_model(args, policy_fn, expert_policy_fn=None):
    print('number of rollouts: ', args.rollouts)

    returns = []
    observations = []
    actions = []
    expert_actions = []

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.env_name)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        print("max steps: ", max_steps)

        for i in range(args.rollouts):
            if args.verbose:
                print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.  # total rewards
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                if expert_policy_fn is not None:
                    expert_action = expert_policy_fn(obs[None, :])
                    expert_actions.append(expert_action)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break

            print('Episode {} finished after {} timesteps'.format(i, steps + 1))
            returns.append(totalr)

        # print('returns; ', returns)
        print('mean return; ', np.mean(returns))
        print('std of return; ', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        if args.policy_type == PolicyType.EXPERT:
            with open(os.path.join(
                    args.expert_data_dir,
                    args.envname + '-' + str(args.num_rollouts) + '.pkl'),
                    'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    return returns, observations, actions, expert_actions
