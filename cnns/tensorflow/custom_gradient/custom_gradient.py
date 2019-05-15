import tensorflow as tf

@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1 - 1 / (1 + e))
    return tf.log(1 + e), grad

def example():
    x = tf.constant(100.)
    y = log1pexp(x)
    dy = tf.gradients(y, x)

    with tf.Session() as sess:
        print("output: ", sess.run(dy))

################################################################################
def my_custom_grad(dy, e):
    return dy * (1 - 1 / (1 + e))

@tf.custom_gradient
def log1pexp2(x):
    e = tf.exp(x)
    def grad(dy):
        return my_custom_grad(dy, e)
    return tf.log(1 + e), grad

def example2():
    x = tf.constant(100.)
    y = log1pexp2(x)
    dy = tf.gradients(y, x)

    with tf.Session() as sess:
        print("output: ", sess.run(dy))


if __name__ == "__main__":
    example2()