import tensorflow as tf

# tensors of rank 0 - scalars
# mammal = tf.Variable("Elephant", tf.string, initializer=)
mammal = tf.get_variable(name="Elephant", shape=(), dtype=tf.string,
                         initializer=tf.constant_initializer("Tony"))
v = tf.get_variable(name="v", shape=(), initializer=tf.zeros_initializer())
w = v + 1
# mammal.initialized_value()
# mammal.assign("Tony")

with tf.Session() as sess:
    # you must explicitly initialize the variables!!!
    tf.global_variables_initializer().run()
    print("w value: ", sess.run(w))
    print("w eval: ", w.eval())
    # print("mammal: ", sess.run(mammal))
    print("evaluate a mammal tensor: ", mammal.eval())

ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.58j, tf.complex64)

squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
print("rank of squares: ", rank_of_squares)

with tf.Session() as sess:
    constant = tf.constant([1, 2, 3])
    tensor = constant * constant
    print(tensor.eval())

with tf.Session() as sess:
    p = tf.placeholder(tf.float32)
    a = tf.Variable(0.0, tf.float32)
    print("tensor a: ", a)
    print("tf.Print(a, [a]): ", tf.Print(a, [a]))
    b = tf.Print(a, [a])
    c = b + 1
    print("c: ", c)
    z = tf.get_variable("z", shape=())
    t = p + 1.0 + a + z
    # t.eval()  # error
    result = t.eval(feed_dict={p: 2.0, a: 20.0, z: 40.0})
    print("eval result: ", result)

value = tf.get_variable("value", shape=(), initializer=tf.zeros_initializer())
assignment = value.assign_add(1)
with tf.control_dependencies([assignment]):
    w = value.read_value()
    print("w value: ", w)


def conv_relu(input, kernel_shape, bias_shape):
    """
    A function to create a convolutional / relu layer.

    :param input:
    :param kernel_shape:
    :param bias_shape:
    :return:
    """
    # Create variable named 'weights'
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases"
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(conv + biases)


input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])


# BELOW: there is a problem with unclear behavior, should we reuse the existing
# tensorflow variables or create new variables?
# x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
# x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])

# !!! All the Variable-s in tensorflow are global
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # variables below will have prefix conv1/
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # variables below will have prefix conv2/
        return conv_relu(relu1, [5, 5, 32, 32], [32])


with tf.Session() as sess:
    # rule: all variables are global in tensorflow
    print("global variables: ", tf.global_variables())
    x = my_image_filter(input1)
    # var = tf.get_variable("conv1/biases")
    # print(var.read_value())
    tf.global_variables_initializer().run()
    print("global variables: ", tf.global_variables())
    # print("local variables: ", tf.local_variables())
    # print("local variables conv1: ", tf.local_variables("conv1/"))
    # tf.global_variables_initializer().run()

    print(sess.run(x))
