# Please check the tf version, as the tf version has been updated
# input_data function will be deprecated in coming version 
# use tf models to this or download the data from other sources

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
n_hidden3 = n_hidden1
n_outputs = n_inputs # same as input

learning_rate = 0.01
l2_reg = 0.0005

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
	a1 = tf.nn.relu(logit_1)

with tf.name_scope('hidden_2'):
	w2_init = initializer([n_hidden1, n_hidden2])
	weights_2 = tf.Variable(w2_init, dtype=tf.float32, name='w_2')
	bias_2 = tf.Variable(tf.zeros([n_hidden2]), name='b_2')

	logit_2 = tf.matmul(a1, weights_2) + bias_2
	a2 = tf.nn.relu(logit_2)

with tf.name_scope('hidden_3'):
	weights_3 = tf.transpose(weights_2, name='w_3')
	bias_3 = tf.Variable(tf.zeros([n_hidden3]), name='b_3')

	logit_3 = tf.matmul(a2, weights_3) + bias_3
	a3 = tf.nn.relu(logit_3)

with tf.name_scope('output'):
	weights_4 = tf.transpose(weights_1, name='w_4')
	bias_4 = tf.Variable(tf.zeros([n_outputs]), name='b_4')

	output = tf.matmul(a3, weights_4) + bias_4


with tf.name_scope('loss'):
	reconstruction_loss = tf.reduce_mean(tf.square(output-X))
	tf.summary.scalar('reconstruction_loss',reconstruction_loss)
	
	regularizer_loss = regularizer(weights_1) + regularizer(weights_2)
	tf.summary.scalar('regularizer_loss',regularizer_loss)

	loss = reconstruction_loss + regularizer_loss
	tf.summary.scalar('total_loss', loss)


with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# training network
init = tf.global_variables_initializer()

epochs = 5
batch_size = 200
loss_list = []
total_loss_list = []

with tf.Session() as session:
	init.run()

	# merge all summaries
	summ = tf.summary.merge_all()

	# write the summaries
	writer = tf.summary.FileWriter(LOGDIR, session.graph)
    
	# save the model for future use
	saver = tf.train.Saver()

	for epoch in range(epochs):
		iterations = mnist.train.num_examples // batch_size
		
		for iteration in range(iterations):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			_, s = session.run([optimizer, summ], feed_dict={X: X_batch})
			total_loss_value, rec_loss_value = session.run([loss, reconstruction_loss], feed_dict={X: X_batch})

			loss_list.append(rec_loss_value)
			total_loss_list.append(total_loss_value)

			if iteration % 100 == 0:
				print('Reconstruction Loss: {}, epoch: {}, iteration: {}'.format(rec_loss_value, epoch+1, iteration))
				print('Total Loss: {}, epoch: {}, iteration: {}'.format(total_loss_value, epoch+1, iteration))

		if epoch==4:
			saver.save(session, os.path.join(LOGDIR, "model.ckpt"), epoch)
			writer.add_summary(s, iteration)

			


