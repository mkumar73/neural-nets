# use pre-build tf estimators to build network, train and test.


"""
tf updates:

tf.examples.tutorials.mnist is deprecated. We will use tf.keras.datasets.mnist instead.
the tf.contrib.learn API was promoted to tf.estimators and tf.feature_columns

tf estimator is one stop shop for building and training ANN.
It is a high level API build on top of tf base class.

"""


import tensorflow as tf

