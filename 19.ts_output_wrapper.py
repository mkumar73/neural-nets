"""
Simple time series prediction using Uni-variate input

Uses the OutputPrjectionWrapper to control the #output
units for RNN. It basically applies fully connected
layer at every time step to reduce the #output as per
requirement.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# network structure
n_inputs = 1
n_outputs = 1
n_steps = 20
n_neurons = 100
lr = 0.0001

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_outputs], name='output')

# basic RNN cell
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
# OutputProjectionWrapper used to control #output units for each RNN cell at each time step
out_wrapper_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=n_outputs)
# normal dynamic rnn cell for output and cell state
output, cell_state = tf.nn.dynamic_rnn(cell=out_wrapper_cell, inputs=X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(output-y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# network structure ends here.

# data preparation starts
t_min, t_max = 0, 30
resolution = 0.1


def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)


def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)


t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="lower right")
plt.xlabel("Time")


# save_fig("time_series_plot")
# plt.show()


# training the network

