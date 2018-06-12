# the stacked AE is trained in different phases
# phase 1 - train the input and the output layer
# phase 2 - train the hidden layer, use the learned weights from phase 1
# phase 3 - stack Phase 1 and phase 2, use the learned weights from phase 1 and phase 2

## AE structure 
# layer - 1 - input layer
# 3 hidden layer
# output layer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# logging
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(old_v)

## log directory
LOGDIR = "graphs/ae/stacked/phasewise/"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

print('Size of dataset:')
print('Training size:{}'.format(len(mnist.train.labels)))
print('Test size:{}'.format(len(mnist.test.labels)))
print('Validation size:{}'.format(len(mnist.validation.labels)))



# # reset tf graph
tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300 
n_hidden2 = 150
n_outputs = n_inputs # same as input

learning_rate = 0.01

# use L2 regularizer
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

# unit variance initializer
initializer = tf.contrib.layers.variance_scaling_initializer()


# create the network structure
with tf.name_scope('input'):
	X = tf.placeholder(tf.float32, shape=[None, n_inputs])


with tf.name_scope('hidden_1'):
	w1_init = initializer([n_inputs, n_hidden1])
	weights_1 = tf.Variable(w1_init, dtype=tf.float32, name='w_1')
	bias_1 = tf.Variable(tf.zeros([n_hidden1]), name='b_1')

	logit_1 = tf.matmul(X, weights_1) + bias_1
	a1 = tf.train.elu(logit_1)

with tf.name_scope('hidden_2'):
	w2_init = initializer([n_hidden1, n_hidden2])
	weights_2 = tf.Variable(w2_init, dtype=tf.float32, name='w_2')
	bias_2 = tf.Variable(tf.zeros([n_hidden2]), name='b_2')

	logit_2 = tf.matmul(a1, weights_2) + bias_2
	a2 = tf.train.elu(logit_2)



