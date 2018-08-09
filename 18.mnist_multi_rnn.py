"""
mnist digit classification using
1. Multi later rnn
2. rnn dyanamic cell
3. Xavier initialization
3. No Dropout
"""

import tensorflow as tf
import numpy as np
import logging
import os

LOGDIR = "graphs/rnn/mnist_multi"


class RnnMnist:

    def __init__(self, session: tf.Session(), data='mnist', n_rnn_cell=75,
                 n_layers=3, lr=0.001, epochs=5, batch_size=128, testing=False):
        """

        :param session:
        :param data:
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
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = 10
        self.testing = testing

    def _load_data(self):
        """

        :return:
        """
        if self.data == 'mnist':
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
        self._data_preprocessing()

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
            x_batch, y_batch = x[batch], y[batch]
            yield x_batch, y_batch

    def rnn_network(self):
        """

        :return:
        """

        # build the network
        with tf.name_scope('input'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_steps, self.n_inputs], name='input')
            y = tf.placeholder(dtype=tf.int64, shape=[None], name='label')

        with tf.variable_scope('rnn', initializer=tf.contrib.layers.xavier_initializer()):

            layers = [tf.contrib.rnn.BasicRNNCell(num_units=self.n_rnn_cell, activation=tf.nn.relu)
                      for layer in range(self.n_layers)]
            multi_layer = tf.contrib.rnn.MultiRNNCell(layers)

            # dropout_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=0.7)

            output, cell_state = tf.nn.dynamic_rnn(multi_layer, X, dtype=tf.float32)
            concat_states = tf.concat(axis=1, values=cell_state)

            tf.summary.histogram('cell state', cell_state)
            tf.summary.histogram('rnn output', output)

        with tf.name_scope('fc'):
            logit = tf.layers.dense(inputs=concat_states, units=self.n_classes, activation=None, name='logit')
            tf.summary.histogram('logit', logit)

        with tf.name_scope('cost'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
            loss = tf.reduce_mean(entropy)
            tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        with tf.name_scope('accuracy'):
            top1_pred = tf.nn.in_top_k(logit, y, 1)
            top1_accuracy = tf.reduce_mean(tf.cast(top1_pred, tf.float32))
            tf.summary.scalar('top1 accuracy', top1_accuracy)

            top2_pred = tf.nn.in_top_k(logit, y, 2)
            top2_accuracy = tf.reduce_mean(tf.cast(top2_pred, tf.float32))
            tf.summary.scalar('top2 accuracy', top2_accuracy)

        # print all trainable variables
        for i in tf.trainable_variables():
            print(i)
        # build process completed

        # load the data
        x_train, x_validation, x_test, y_train, y_validation, y_test = self._train_validation_split()

        # training process started
        init = tf.global_variables_initializer()

        self.session.run(init)

        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOGDIR, self.session.graph)
        saver = tf.train.Saver()

        print('Training process started:........')
        for epoch in range(self.epochs):
            for x_batch, y_batch in self.next_batch(x_train, y_train, self.batch_size):
                x_batch = x_batch.reshape((-1, self.n_steps, self.n_inputs))
                s, _, loss_value = self.session.run([summ, optimizer, loss],
                                                    feed_dict={X: x_batch, y: y_batch})
                # add summary for every iteration
                writer.add_summary(s, epoch)

            batch_top1, batch_top2, batch_loss = self.session.run([top1_accuracy, top2_accuracy, loss],
                                                                  feed_dict={X: x_batch, y: y_batch})
            x_validation = x_validation.reshape((-1, self.n_steps, self.n_inputs))
            val_top1, val_top2, val_loss = self.session.run([top1_accuracy, top2_accuracy, loss],
                                                            feed_dict={X: x_validation, y: y_validation})

            print('Epoch:{0}, batch_top1_acc:{1}, batch_top2_acc:{2}, loss:{3}'
                  .format(epoch+1, batch_top1, batch_top2, batch_loss))
            print('Epoch:{0}, val_top1_acc:{1}, val_top2_acc:{2}, loss:{3}'
                  .format(epoch+1, batch_top1, batch_top2, val_loss))

            if epoch == self.epochs-1:
                saver.save(self.session, os.path.join(LOGDIR, 'model.ckpt'), epoch)

        if self.testing:
            with tf.Session() as session:
                saver = tf.train.Saver()
                saver.restore(session, tf.train.latest_checkpoint(LOGDIR))

                count = 0
                iterations = 5
                test_top1_acc = []
                test_top2_acc = []
                test_cost = []
                for _iter in range(iterations):
                    x_test = x_test.reshape((-1, self.n_steps, self.n_inputs))
                    top1_test_acc, top2_test_acc, test_loss\
                        = session.run([top1_accuracy, top2_accuracy, loss], feed_dict={X: x_test, y: y_test})
                    test_top1_acc.append(top1_test_acc)
                    test_top2_acc.append(top2_test_acc)
                    test_cost.append(test_loss)
                    count += 1

            # print cost and accuracy calculated for test set
            print("top1_test_accuracy:{} ".format(np.sum(test_top1_acc) / count))
            print("top2_test_accuracy:{} ".format(np.sum(test_top2_acc) / count))
            print("average test loss:{} ".format(np.sum(test_cost) / count))

        return


def main():
    tf.reset_default_graph()

    with tf.Session() as session:
        rnn_mnist = RnnMnist(session, data='mnist', n_rnn_cell=100, n_layers=3,
                             lr=0.0001, epochs=10, testing=True)
        rnn_mnist.rnn_network()


if __name__ == '__main__':

    main()
