import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random_uniform([32, 3, 4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset1.batch(16)

print("ds shapes after batching: ",
      dataset1.output_shapes)  # ==> "(10,)"
