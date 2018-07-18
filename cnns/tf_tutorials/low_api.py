from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

print("tensorflow version: ", tf.__version__)

a = tf.constant(3.0, dtype=tf.float32, name="const0")
b = tf.constant(4.0, name="const1")  # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

writer = tf.summary.FileWriter('events/')
writer.add_graph(tf.get_default_graph())

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

my_data = [
    [0, 1, ],
    [2, 3, ],
    [4, 5, ],
    [6, 7, ],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

x_input = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y_output = linear_model(x_input)
init = tf.global_variables_initializer()

# feature columns

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
with tf.Session() as sess:
    sess.run((var_init, table_init))

with tf.Session() as session:
    result = session.run(total)
    print("type of the result: ", type(result))
    print("result: ", result)
    print(session.run({'ab': (a, b), 'total': total}))
    print("vec: ", session.run(vec))
    # probably we will get different values because of randomness
    print("vec: ", session.run(vec))
    print(session.run({"out1, out2: ": (out1, out2)}))
    # using the feed_dict argument of the run method to feed concrete
    # values to the placeholders
    print("z: ", session.run(z, feed_dict={x: 3, y: 4.5}))
    print("z2: ", session.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
    for elem in my_data:
        print("raw elem: ", elem)
        print("next item: ", session.run(next_item))

    # restart the iterator
    next_item = slices.make_one_shot_iterator().get_next()
    while True:
        try:
            print("next item in the while loop: ",
                  session.run(next_item))
        except tf.errors.OutOfRangeError:
            print("out of range")
            break

    # initialize the dense layer
    session.run(init)
    print("y_output: ",
          session.run(y_output, feed_dict={x_input: [[1, 2, 3]]}))
    print("y_output 2: ",
          session.run(y_output,
                      feed_dict={x_input: [[1, 2, 3], [4, 5, 6]]}))
    session.close()
