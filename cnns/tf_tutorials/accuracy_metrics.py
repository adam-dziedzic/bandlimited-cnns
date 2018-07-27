import tensorflow as tf

logits = tf.placeholder(tf.int64, [2, 3])
labels = tf.Variable([[0, 1, 0], [1, 0, 1]])

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                  predictions=tf.argmax(logits, 1))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    stream_vars = [i for i in tf.local_variables()]
    print(stream_vars)

    # print(sess.run([acc, acc_op], feed_dict={logits: [[1, 2, 3], [4, 5, 6]]}))
    print("acc_op1:", sess.run(acc_op, feed_dict={logits:[[0,1,0],[1,0,1]]}))
    print(sess.run([acc]))
    print("acc_op2:", sess.run([acc_op], feed_dict={logits: [[1,1,0],[0,0,0]]}))
