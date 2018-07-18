"""
class based approach to for MNIST classification
using tf.layers.dense instead of contrib
"""

import tensorflow as tf
import numpy as np
import  logging
tf.logging.set_verbosity(tf.logging.INFO)

class MNISTDENSE():

    def __init__(self, session:tf.Session(), data='mnist', input_size=28, lr=0.01,
                 batch_size=64, epochs=10):
        """
        :param session: tf session
        :param data: name of dataset
        :param input_size: input size of image
        :param lr: learning rate
        :param batch_size: batch size
        :param epochs: number of epochs to train on
        """

        self.session = session
        self.data = data
        self.input_size = input_size
        self.image_size = input_size*input_size
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def _load_data(self):
        """
        :return: load the data from dataset library
        """
        self.data.lower()
        if self.data=='mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        else:
            logging.error('Dataset error: Only implmented for MNIST as of now.!!')
        return

    def _data_preprocessing(self):
        """
        private function
        :return: processed data
        """
        self._load_data()

        x_train = self.x_train.astype(np.float32).reshape(-1, self.image_size)/255.0
        x_test = self.x_test.astype(np.float32).reshape(-1, self.image_size)/255.0

        y_train = self.y_train.astype(np.int64)
        y_test = self.y_test.astype(np.int64)
        return x_train, x_test, y_train, y_test

    def _train_test_split(self, _index = 5000):
        """

        :param _index: range of trainig and validation data
        :return: train, validation and test set
        """
        x_train, x_test, y_train, y_test = self._data_preprocessing()

        x_train, x_validation = x_train[5000:], x_train[:5000]
        y_train, y_validation = y_train[5000:], y_train[:5000]
        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def shuffle_batch(self, x, y, batch_size):
        """
        :param x: image
        :param y: labels
        :param batch_size: #samples in a batch
        :return: shuffeld samples, images and labels
        """
        rnd_idx = np.random.permutation(len(x))
        n_batches = len(x) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            x_batch, y_batch = x[batch_idx], y[batch_idx]
            yield x_batch, y_batch

    def check_sample_data(self):
        """
        :return: print something
        """
        train, val, test, _, _, _ = self._train_test_split(_index=5000)
        print('Size of train, validation and test set:\n')
        print(train.shape)
        print(val.shape)
        print(test.shape)
        print('Sample data:\n')
        print(train[:10])
        return

    def build_network(self, session, n_h1, n_h2, n_output):
        """

        :param session: tensorflow session
        :param n_h1: #neurons for h1
        :param n_h2: #neurons for h2
        :param n_output: #neurons for output layer
        :return: build and train the network
        """

        X = tf.placeholder(tf.float32, shape=(None, self.image_size), name='X')
        y = tf.placeholder(tf.int64, shape=(None), name='y')

        with tf.name_scope('fully_connected'):
            h1 = tf.layers.dense(X, n_h1, activation=tf.nn.relu, name='hidden_1' )
            h2 = tf.layers.dense(h1, n_h2, activation=tf.nn.relu, name='hidden_2')
            logits = tf.layers.dense(h2, n_output, name='output')

            # y_pred = tf.nn.softmax(logits)

        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(entropy, name='loss')

        # implement gradient clipping, it might not improve the performance
        # but its important to know the implementation technique
        with tf.name_scope('optimize'):
            threshold = 1.0
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grad_var = optimizer.compute_gradients(loss)
            clipped_grads = [(tf.clip_by_value(grad, -threshold, threshold), var)
                             for grad, var in grad_var]
            training_op = optimizer.apply_gradients(clipped_grads)

        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        x_train, x_validation, x_test, y_train, y_validation, y_test = self._train_test_split()

        init = tf.global_variables_initializer()
        session.run(init)

        for epoch in range(self.epochs):
            for x_batch, y_batch in self.shuffle_batch(x_train, y_train, self.batch_size):
                session.run(training_op, feed_dict={X: x_batch, y: y_batch})
            acc_batch = session.run(accuracy, feed_dict={X: x_batch, y: y_batch})
            acc_val = session.run(accuracy, feed_dict={X: x_validation, y: y_validation})

            print('Epoch:', epoch, 'Batch accuracy:', acc_batch, 'Validation accuracy:', acc_val)
            # tf.logging.info(acc_batch)
            # tf.logging.info(acc_val)

        return


def main():

    with tf.Session() as session:
        mnist = MNISTDENSE(session, 'mnist', 28, 0.01, 64, 10)
        mnist.build_network(session, n_h1=25, n_h2=25, n_output=10)


if __name__ == '__main__':
    main()

