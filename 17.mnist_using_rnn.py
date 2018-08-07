"""
mnist digit classification using rnn dyanamic cell
"""

import tensorflow as tf
import numpy as np
import logging

LOGDIR = "graphs/rnn/mnist"


class RnnMnist():

    def __init__(self, dataset='mnist', n_inputs=28, n_steps=28, n_rnn_cell=75,
                 lr=0.001, epochs=5, batch_size=128):
        """

        :param dataset:
        :param n_inputs:
        :param n_steps:
        :param n_rnn_cell:
        :param lr:
        :param epochs:
        :param batch_size:
        """
        self.dataset = dataset.lower()
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_rnn_cell = n_rnn_cell
        self.lr =lr
        self.epochs = epochs
        self.batch_size = batch_size


    def _load_data(self):
        """

        :return:
        """
        if self.dataset=='mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        else:
            logging.error('Enter the correct name for the dataset..!!')
        return

    def _data_preprocessing(self):

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

        with tf.name_scope('input'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_steps, self.n_inputs], name='input')
            y = tf.placeholder(dtype=tf.int64, shape=[None], name='label')




