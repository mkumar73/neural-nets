# very basic implementation of RNN using low level tf API

## Low-level API implementation
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_neurons = 5
n_inputs = 3

g1 = tf.Graph()
with g1.as_default() as g1:
    X0 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    X1 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])

    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    b = tf.Variable(tf.zeros(shape=[1, n_neurons], dtype=tf.float32))

    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
    Y2 = tf.tanh(tf.matmul(Y1, Wy) + tf.matmul(X2, Wx) + tf.matmul(X0, Wx) + b)

    init = tf.global_variables_initializer()

    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
    X2_batch = np.array([[2, 3, 4], [6, 7, 9], [5, 7, 8], [1, 2, 1]]) # t = 2


with tf.Session(graph=g1) as sess:
    init.run()
    Y0_val, Y1_val, Y2_val = sess.run([Y0, Y1, Y2], feed_dict={X0: X0_batch, X1: X1_batch, X2: X2_batch})

print('----------------------------------')
print('Y0 value for the inputs:\n',Y0_val)
print('Shape of Y0_val:\n', Y0_val.shape)
print()
print('Y1 value for the inputs:\n', Y1_val)
print('Shape of Y1_val:\n', Y1_val.shape)
print()
print('Y2 value for the inputs:\n', Y2_val)
print('Shape of Y2_val:\n', Y2_val.shape)
print('----------------------------------')


###########################################################
# Use of RNN cell in high level API
# Using BasicRNNCell and static_rnn

tf.reset_default_graph()

g2 = tf.Graph()
with g2.as_default() as g2:
    X0 = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    X1 = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    basic_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seq, cell_state = tf.contrib.rnn.static_rnn(basic_rnn_cell, [X0, X1, X2], dtype=tf.float32)

    Y0, Y1, Y2 = output_seq

    init = tf.global_variables_initializer()

    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
    X2_batch = np.array([[2, 3, 4], [6, 7, 9], [5, 7, 8], [1, 2, 1]]) # t = 2


with tf.Session(graph=g2) as session:
    init.run()
    Y0_val, Y1_val, Y2_val = session.run([Y0, Y1, Y2], feed_dict={X0: X0_batch, X1: X1_batch, X2: X2_batch})

print('----------------------------------')
print('Y0 value for the inputs:\n',Y0_val)
print('Shape of Y0_val:\n', Y0_val.shape)
print()
print('Y1 value for the inputs:\n', Y1_val)
print('Shape of Y1_val:\n', Y1_val.shape)
print()
print('Y2 value for the inputs:\n', Y2_val)
print('Shape of Y2_val:\n', Y2_val.shape)
print('----------------------------------')

