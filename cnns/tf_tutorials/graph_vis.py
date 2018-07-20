import tensorflow as tf

# Build your graph.
dtype = tf.float32
x = tf.constant([[37.0, -23.0], [1.0, 4.0]], dtype=dtype)
w = tf.Variable(tf.random_uniform(shape=[2, 2], dtype=dtype))
y = tf.matmul(x, w)
y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # `sess.graph` provides access to the graph used in a
    # <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
    writer = tf.summary.FileWriter("graph_vis", sess.graph)

    sess.run(tf.global_variables_initializer())
    # Perform your computation...
    for i in range(10000):
        _, loss_value = sess.run((train_op, loss))
        # print("loss value: ", loss_value)

    print("prediction after training: ", sess.run(y))
    writer.close()

    """
    Now, in a new terminal, launch TensorBoard with the following shell command:

    > tensorboard --logdir .
    """
