# use pre-build tf estimators to build network, train and test.

"""
tf updates:

tf.examples.tutorials.mnist is deprecated. We will use tf.keras.datasets.mnist instead.
the tf.contrib.learn API was promoted to tf.estimators and tf.feature_columns

tf estimator is one stop shop for building and training ANN.
It is a high level API build on top of tf base class.

"""


import tensorflow as tf
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32).reshape(-1, 28*28) / 255.0
x_test = x_test.astype(np.float32).reshape(-1, 28*28) / 255.0

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# print(x_train[:5])
# print(y_train[:5])

x_train, x_validation = x_train[5000:], x_train[:5000]
y_train, y_validation = y_train[5000:], y_train[:5000]

# print(len(x_train), len(x_validation), len(x_test))
# print(len(y_train), len(y_validation), len(y_test))

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

dnn_clf = tf.estimator.DNNClassifier(hidden_units=[25, 25], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"X": x_train}, y=y_train, num_epochs=20, batch_size=50, shuffle=True)

dnn_clf.train(input_fn=input_fn)


test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"X": x_test}, y=y_test, shuffle=False)

eval_results = dnn_clf.evaluate(input_fn=test_input_fn)

print(eval_results)


