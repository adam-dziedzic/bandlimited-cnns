import numpy as np
import tensorflow as tf

# Create a variable.
array_np = np.array([1.0, 2.0, 3.0])
array_tensor = tf.convert_to_tensor(array_np)
w = tf.Variable(array_tensor)

# Assign a new value to the variable with `assign()` or a related method.
w.assign(array_tensor)

gradient = np.array([0.1, 0.2, 0.3])
gradient = tf.convert_to_tensor(gradient)
w.assign_add(gradient)

with tf.Session() as sess:
    # Run the variable initializer.
    sess.run(w.initializer)
    print(sess.run(w))
    print(sess.run(w))
