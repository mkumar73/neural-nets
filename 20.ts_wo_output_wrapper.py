"""
Simple time series prediction using Uni-variate input

Without OutputProjectionWrapper
Only 1 fully connected layer is required for
all the time step instead of fc for every time step.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

LOGDIR = "graphs/rnn/ts_wo_wrapper"

# network structure
n_inputs = 1
n_outputs = 1
n_steps = 20
n_neurons = 200
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
init = tf.global_variables_initializer()
n_iterations = 1000
batch_size = 64

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    for iteration in range(n_iterations+1):
        # call the next_batch function
        # keep in mind, its not a generator
        x_batch, y_batch = next_batch(batch_size, n_steps)
        session.run(optimizer, feed_dict={X: x_batch, y: y_batch})

        if iteration % 100 == 0:
            mse_value = session.run(loss, feed_dict={X: x_batch, y: y_batch})
            print('MSE for iteration: {}, is: {}'.format(iteration, mse_value))
        if iteration == n_iterations:
            saver.save(session, os.path.join(LOGDIR, 'model.ckpt'), iteration)


with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint(LOGDIR))
    x_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = session.run(output, feed_dict={X: x_new})

inp_out = np.c_[x_new, y_pred]
print('Input and output pair:\n'.format(inp_out))

plt.title("Model Prediction", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0, :, 0], "r.", markersize=10, label="prediction")
plt.legend(loc="lower right")
plt.xlabel("Time")

# save_fig("time_series_pred_plot")
plt.show()