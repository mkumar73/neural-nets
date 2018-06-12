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

# create the network structure




