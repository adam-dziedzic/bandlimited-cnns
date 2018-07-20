import tensorflow as tf

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)

with tf.Session() as sess:
    # initialize variables in the TensorFlow graph
    init = tf.global_variables_initializer()
    print("global variables: ", tf.global_variables())
    sess.run(init)
    result = sess.run(y_pred)
    print("result: ", result)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print(sess.run(loss))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    for i in range(1000):
        _, loss_value = sess.run((train, loss))
        # print("loss value: ", loss_value)

    print("prediction after training: ", sess.run(y_pred))

