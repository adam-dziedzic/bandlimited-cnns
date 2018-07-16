import sys

import argparse
import iris_data
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int,
                    help="batch size")

args = parser.parse_args(sys.argv[1:])
print("given or default batch size: ", args.batch_size)

train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train
print("len of mnist_y: ", len(mnist_y))
print("type of mnist_x: ", type(mnist_x))
print("shape of mnist_x: ", mnist_x.shape)

mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print("mnist_ds: ", mnist_ds)
print("type of mnist_ds: ", type(mnist_ds))

# add batching to the dataset
# the dataset has an unknown batch size because the last batch will
# have fewer elements
print(mnist_ds.batch(args.batch_size))

## handle CSV data

"""Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
train_path, test_path = iris_data.maybe_download()
ds = tf.data.TextLineDataset(train_path).skip(1)
print("ds: ", ds)

# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    # print("fields.shape: ", fields.shape)
    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))

    # Separate the label from the features
    label = features.pop('label')

    return features, label


# with tf.Session() as sess:
#     for line in ds:
#         features, label = _parse_line(line)
#         print("features: ", features)
#         print("label: ", label)

ds = ds.map(_parse_line)
print(ds)

train_path, test_path = iris_data.maybe_download()

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)
print("Train the estimator")
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda: iris_data.csv_input_fn(train_path, batch_size))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = est.predict(
    input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                             labels=None,
                                             batch_size=args.batch_size))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))

print("Evaluate the model.")
(train_x, train_y), (test_x, test_y) = iris_data.load_data()
eval_result = est.evaluate(
    input_fn=lambda: iris_data.eval_input_fn(test_x, test_y,
                                             args.batch_size))
# eval_result = est.evaluate(input_fn=lambda: iris_data.csv_input_fn(test_path, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

print("Load the cifar-10 dataset")
(x_train, y_train), (
    x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("len of x_train cifar-10: ", len(x_train))
print("type of x_train cifar-10: ", type(x_train))
print("shape of x_train cifar-10: ", x_train.shape)

# cifar data set
cifar_ds = tf.data.Dataset.from_tensor_slices(x_train)
print("cifar_ds: ", cifar_ds)
print("type of cifar_ds: ", type(cifar_ds))

# add batching to the dataset
# the dataset has an unknown batch size because the last batch will
# have fewer elements
print("cifar output shapes before batching: ", cifar_ds.output_shapes)
print("cifar-10 add batching: ", cifar_ds.batch(args.batch_size))

print("cifar output types: ", cifar_ds.output_types)
print("cifar output shapes: ", cifar_ds.output_shapes)
