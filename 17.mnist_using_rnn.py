"""
mnist digit classification using rnn dyanamic cell
"""

import tensorflow as tf
import numpy as np
import logging
import os

LOGDIR = "graphs/rnn/mnist"


class RnnMnist():

    def __init__(self, session: tf.Session(), data='mnist', n_rnn_cell=75,
                 lr=0.001, epochs=5, batch_size=128):
        """

        :param session:
        :param dataset:
        :param n_inputs:
        :param n_steps:
        :param n_rnn_cell:
        :param lr:
        :param epochs:
        :param batch_size:
        """
        self.session = session
        self.data = data.lower()
        self.n_inputs = 28
        self.n_steps = 28
        self.n_rnn_cell = n_rnn_cell
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = 10



    def _load_data(self):
        """

        :return:
        """
        if self.data=='mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        else:
            logging.error('Enter the correct name for the dataset..!!')
        return

    def _data_preprocessing(self):
        """

        :return:
        """
        self._load_data()

        self.x_train = self.x_train.astype(np.float32)/255.0
        self.x_test = self.x_test.astype(np.float32)/255.0

        self.y_train = self.y_train.astype(np.int64)
        self.y_test = self.y_test.astype(np.int64)
        return

    def _train_validation_split(self, _index=5000):
        """

        :param _index:
        :return:
        """
        x_train, x_validation = self.x_train[_index:], self.x_train[:_index]
        y_train, y_validation = self.y_train[_index:], self.y_train[:_index]
        x_test, y_test = self.x_test, self.y_test
        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def next_batch(self, x, y, batch_size):
        """

        :param x:
        :param y:
        :param batch_size:
        :return:
        """
        rnd_idx = np.random.permutation(len(x))
        n_batches = len(x) // batch_size
        for batch in np.array_split(rnd_idx, n_batches):
            x_batch, y_batch = x[batch, :, :, :], y[batch]
            yield x_batch, y_batch

    def rnn_network(self):
        """

        :return:
        """

        # build the network
        with tf.name_scope('input'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_steps, self.n_inputs], name='input')
            y = tf.placeholder(dtype=tf.int64, shape=[None], name='label')

        with tf.name_scope('rnn_cell'):
            basic_cell = tf.contrib.rnn.BasicRNNCell(n_units=self.n_rnn_cell)
            output, cell_state = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

        with tf.name_scope('fc'):
            logit = tf.layers.dense(inputs=cell_state, units=self.n_classes, activation=None, name='logit')

        with tf.name_scope('cost'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
            loss = tf.reduce_mean(entropy)
            tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        with tf.name_scope('accuracy'):
            prediction = tf.nn.in_top_k(logit, y, 1)
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        # build process completed

        # load the data
        x_train, x_validation, x_test, y_train, y_validation, y_test = self._train_validation_split()

        # training process started
        tf.reset_default_graph()
        init = tf.global_variables_initializer()

        self.session.run(init)

        summ = self.session.merge_all()
        writer = tf.summary.FileWriter(LOGDIR, self.session.graph)
        saver = tf.train.Saver()

        for epoch in range(self.epochs):
            for x_batch, y_batch in self.next_batch(x_train, y_train, self.batch_size):
                x_batch = x_batch.reshape((-1, self.n_steps, self.n_inputs))
                _, _, loss_value = self.session.run([summ, optimizer, loss],
                                                    feed_dict={X: x_batch, y: y_batch})

            batch_accuracy = self.session.run(accuracy, feed_dict={X: x_batch, y: y_batch})
            x_validation = x_validation.reshape((-1, self.n_steps, self.n_inputs))
            val_accuracy = self.session.runs(accuracy, feed_dict={X: x_validation, y: y_validation})

            print('Epoch:{}, Batch Accuracy:{}, Validation Accuracy:{}',format(epoch, batch_accuracy, val_accuracy))

            writer.add_summary(summ, epoch)

            if epoch == self.epochs-1:
                saver.save(self.session, os.path.join(LOGDIR, 'model.ckpt'), epoch)

            return


def main():

    with tf.Session() as session:
        rnn_mnist = RnnMnist(session, data='mnist', n_rnn_cell=75)
        rnn_mnist.rnn_network()

if __name__=='__main__':
    main()














