# linear autoencoder works as PCA to decompose the higher dimension data into lower dimension

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


LOGDIR = "graphs/ae/linear/"
# seed
np.random.seed(4)

# prepare the input data
rows = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(rows) * 3 * np.pi / 2 - 0.5
data = np.empty((rows, 3))
print(data.shape)

data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(rows) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(rows) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(rows)

# print sample data
print('Unprocessed sample data:')
print(data[:5,:])

# normalize the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

print('Processed training data:')
print(X_train[:5, :])


# reset tf graph
tf.reset_default_graph()

n_inputs = 3
n_hidden = 2 # encoding layer
n_outputs = n_inputs # same as input

learning_rate = 0.01

# create the network structure

#input
with tf.name_scope('inputs'):
	X = tf.placeholder(tf.float32, shape=[None, n_inputs])

# hidden layer
with tf.name_scope('hidden'):
	hidden = tf.layers.dense(X, n_inputs)

# output layer
with tf.name_scope('output'):
	output = tf.layers.dense(hidden, n_outputs)

# reconstruction loss
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(output - X))
	tf.summary.scalar('loss', loss)

# optmizer, Adam is used
with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# training network
init = tf.global_variables_initializer()

n_iterations = 1000
encoding = hidden
loss_list = []

with tf.Session() as session:
	init.run()

	# merge all summaries
	summ = tf.summary.merge_all()

	# write the summaries
	writer = tf.summary.FileWriter(LOGDIR, session.graph)
    
	# save the model for future use
	saver = tf.train.Saver()

	for iteration in range(n_iterations):
		loss_value, _, s = session.run([loss, optimizer, summ], feed_dict={X: X_train})

		loss_list.append(loss_value)

		if iteration==999:
			saver.save(session, os.path.join(LOGDIR, "model.ckpt"), iteration)
			writer.add_summary(s, iteration)

	encoding_values = encoding.eval(feed_dict={X: X_test})


# plot the result of encoded values of test data
# the plot looks like the PCA decomposition into two PC's 
# that accounts for the most variation explained by the data.
fig = plt.figure()
plt.plot(encoding_values[:,0], encoding_values[:, 1], "b.")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.show()


# plot loss
fig = plt.figure()
plt.plot(loss_list, "b.")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()


