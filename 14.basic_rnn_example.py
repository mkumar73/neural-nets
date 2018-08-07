# very basic implementation of RNN using low level tf API

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_neurons = 5
n_inputs = 3

X0 = tf.placeholder(dtype=tf.float32, shape=[None, 3])
X1 = tf.placeholder(dtype=tf.float32, shape=[None, 3])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print('Y0 value for the inputs:\n',Y0_val)
print('Shape of Y0:\n', Y0_val.shape)
print()
print('Y1 value for the inputs:\n', Y1_val)
print('Shape of Y0:\n', Y1_val.shape)
