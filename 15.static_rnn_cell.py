import tensorflow as tf
import numpy as np

###########################################################
# Use of RNN cell in high level API
# Using BasicRNNCell and static_rnn
# Packaged input sequence for RNN

LOGDIR = "graphs/rnn/static"

tf.reset_default_graph()

n_steps = 3
n_inputs = 3
n_neurons = 5

g3 = tf.Graph()
with g3.as_default() as g3:
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

    basic_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seq, cell_state = tf.contrib.rnn.static_rnn(basic_rnn_cell, X_seqs, dtype=tf.float32)

    output = tf.transpose(tf.stack(output_seq), perm=[1, 0, 2])

    init = tf.global_variables_initializer()
    X_batch = np.array([
        # t = 0      # t = 1    # t = 3
        [[0, 1, 2], [9, 8, 7], [2, 3, 4]], # instance = 1
        [[3, 4, 5], [0, 0, 0], [6, 7, 9]], # instance = 1
        [[6, 7, 8], [6, 5, 4], [5, 7, 8]], # instance = 1
        [[9, 0, 1], [3, 2, 1], [1, 2, 1]]  # instance = 1
    ])


with tf.Session(graph=g3) as session:
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(session.graph)
    init.run()
    output_val = session.run(output, feed_dict={X: X_batch})

print('----------------------------------')
print('Output_val for the packaged inputs:\n', output_val)
print('Shape of Output_val:\n', output_val.shape)
print()
print(np.transpose(output_val, axes=[1, 0, 2])[1])

