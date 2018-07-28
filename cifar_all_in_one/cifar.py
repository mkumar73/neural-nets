"""
The exercise is to implement CIFAR object classification from scratch.
We will use the CIFAR10 dataset from tf.keras.datasets.cifar10.
Main idea is to build a CNN to classify correctly the images.
We will try to use all the applicable optimization techniques as mentioned below:
1. Gradient descent variants: Adam, RMSProp, SGD, SGD+Momentum etc.
2. Weight initialization: He initialization, Xaviers initialization
3. Batch Normalization (BN)
4. Early Stopping
5. Dropouts
6. and so on.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging


LOGDIR = "logs/"


class CIFAR10():

    def __init__(self, session: tf.Session(), data='cifar', lr=0.01,
                 batch_size=64, epochs=5, early=False, optimizer='adam',
                 bn=False, init=None):
        """

        :param session: tf session
        :param data: name of dataset
        :param lr: learning rate
        :param batch_size: #sample per batch
        :param epochs: #epoch
        :param early: early stopping flag
        :param optimizer: optimizer name
        :param bn: batch normalization
        :param init: initialization parameter
        """
        self.session = session
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early = early
        self.optimizer = optimizer
        self.bn = bn
        self._init = init

    def _load_data(self, data):
        """
        :param data: dataset name
        :return: dataset and labels for training and test
        """
        data.lower()
        if data == 'cifar':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        else:
            logging.error('Incorrect dataset name given')
        return

    def _data_preprocessing(self):
        """
        private function
        :return: processed data
        """
        self._load_data('cifar')

        self.x_train = self.x_train.astype(np.float32) # .reshape(-1, self.image_size)/255.0
        self.x_test = self.x_test.astype(np.float32) # .reshape(-1, self.image_size)/255.0

        self.y_train = self.y_train.astype(np.int64).reshape(-1, )
        self.y_test = self.y_test.astype(np.int64).reshape(-1, )
        return # x_train, x_test, y_train, y_test

    def _train_val_split(self, _index=5000):
        """

        :param _index: index for slicing
        :return: training, validation and test set data
        """
        self._data_preprocessing()

        x_train, x_validation = self.x_train[5000:], self.x_train[:5000]
        y_train, y_validation = self.y_train[5000:], self.y_train[:5000]
        x_test, y_test = self.x_test, self.y_test
        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def check_sample_data(self):
        """
        :return: print datasets size
        """
        train, val, test, y_train, y_val, y_test = self._train_val_split(_index=5000)
        print('Size of train, validation and test set:\n')
        print('Training data size:', train.shape)
        print('Validation data size:', val.shape)
        print('Test data size:', test.shape)
        print('Test label shape:', y_test.shape)
        # print('Sample data:\n')
        # print(train[:10])
        return

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

    def data_investigation(self, x, y, show=False):
        """
        function to investigate data using plots
        :param show: plot or not
        :param x: subplot parameter
        :param y: subplot parameter
        :return: plot figures with labels from training data
        """
        n_images = x*y

        x_train, x_validation, x_test, y_train, y_validation, y_test = self._train_val_split()

        sample_image = x_train[:n_images]
        sample_label = y_train[:n_images]

        label_to_word = {
            0: "Airplane",
            1: "Autombile",
            2: "Bird",
            3: "Cat",
            4: "Deer",
            5: "Dog",
            6: "Frog",
            7: "Horse",
            8: "Ship",
            9: "Truck"
        }

        fig, axs = plt.subplots(x, y)
        for i, ax in enumerate(np.reshape(axs, [-1])):
            # ax.imshow(sample_image[i, :, :, :])
            ax.imshow(sample_image[i])
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            ax.axis('off')
            ax.set_title(str(int(sample_label[i])) + ':' + label_to_word[int(sample_label[i])])
            if show:
                plt.show()
        return

    def fully_connected(self, input, kernel_shape, act_fn='relu', name='fc', output=False):
        """

        :param input: input
        :param kernal_shape: matrix shape
        :param act_fn: activation function
        :param name: name of layer
        :param output: logit or output
        :return: activated output or logit
        """
        with tf.variable_scope(name):
            if self._init == 'xavier':
                init = tf.contrib.layers.xavier_initializer
            else:
                init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name='weight', shape=kernel_shape, initializer=init)
            biases = tf.get_variable(name='bias', shape=kernel_shape[-1], initializer=init)

            fc = tf.matmul(input, weights)

            if not output:
                if act_fn == 'relu':
                    return tf.nn.relu(fc + biases)
                else:
                    return tf.nn.tanh(fc + biases)
            else:
                return fc + biases

    def conv_relu(self, input, filter_shape, bias_shape, name='conv', is_weights=False):
        """

        :param input: input
        :param filter_shape: filter shape
        :param bias_shape: bias shape
        :param name: name
        :param is_weights: if weights are required for visualization
        :return: convolved result
        """
        with tf.variable_scope(name):
            if self._init == 'xavier':
                init = tf.contrib.layers.xavier_initializer
            else:
                init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name='weight', shape=filter_shape, initializer=init)
            biases = tf.get_variable(name='bias', shape=bias_shape, initializer=init)

            conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

            if not is_weights:
                return tf.nn.relu(conv + biases)
            else:
                return tf.nn.relu(conv + biases), weights

    def max_pooling(self, input, name='maxpool'):
        """

        :param input: input
        :param name: name
        :return: pooled result
        """
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(self):
        # TODO: complete function definition
        return

    def build_and_train(self):
        """
        :return:
        """

        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input')
            y = tf.placeholder(tf.int64, shape=[None], name='label')

        with tf.name_scope('conv1'):
            conv1, conv1_w = self.conv_relu(X, [3, 3, 3, 16], [16], name='conv1', is_weights=True)
            tf.summary.histogram('conv1', conv1)

            pool1 = self.max_pooling(conv1, name='pool1')
            tf.summary.histogram('pool1', pool1)

        with tf.name_scope('conv1'):
            conv2, conv2_w = self.conv_relu(conv1, [3, 3, 16, 32], [32], name='conv2', is_weights=True)
            tf.summary.histogram('conv2', conv2)

            pool2 = self.max_pooling(conv2, name='pool2')
            tf.summary.histogram('pool2', pool2)

        # with tf.name_scope('conv3'):
        #     conv3, conv3_w = self.conv_relu(conv2, [3, 3, 32, 64], [64], name='conv3', is_weights=True)
        #     tf.summary.histogram('conv3', conv3)
        #
        #     pool3 = self.max_pooling(conv3, name='pool3')
        #     tf.summary.histogram('pool3', pool3)
        with tf.name_scope('flatten'):
            fc_input = tf.reshape(pool2, [-1, 8 * 8 * 32])

        with tf.name_scope('fc'):
            fc = self.fully_connected(fc_input, [8 * 8 * 32, 64], act_fn='relu', name='fc1', output=False)
            tf.summary.histogram('fc', fc)

        with tf.name_scope('logit'):
            logit = self.fully_connected(fc, [64, 10], act_fn='relu', name='output', output=True)
            tf.summary.histogram('output', logit)

        # print all trainable variables
        for i in tf.trainable_variables():
            print(i)

        with tf.name_scope('prediction'):
            # normalize the probabolity value so sum upto 1 for each row.
            y_pred = tf.nn.softmax(logit)
            # get the predicted class for each sample using argmax
            y_pred_cls = tf.argmax(y_pred, axis=1)

        with tf.name_scope('cost'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y)
            cost = tf.reduce_mean(entropy)
            tf.summary.scalar('cost', cost)

        with tf.name_scope('optimizer'):
            self.optimizer.lower()
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(cost)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)

        with tf.name_scope('accuracy'):
            # performance measures
            correct_prediction = tf.equal(y_pred_cls, y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        x_train, x_validation, x_test, y_train, y_validation, y_test = self._train_val_split()

        init = tf.global_variables_initializer()
        self.session.run(init)

        # merge all summaries
        summ = tf.summary.merge_all()

        # write the summaries
        writer = tf.summary.FileWriter(LOGDIR, self.session.graph)

        # save the model for future use
        saver = tf.train.Saver()

        for epoch in range(self.epochs):
            for x_batch, y_batch in self.shuffle_batch(x_train, y_train, self.batch_size):
                _, s = self.session.run([optimizer, summ], feed_dict={X: x_batch, y: y_batch})

            batch_accuracy = self.session.run(accuracy, feed_dict={{X: x_batch, y: y_batch}})
            validation_accuracy = self.session.run(accuracy, feed_dict={X: x_validation, y: y_validation})

            print('Epoch:', epoch, 'Batch accuracy:', batch_accuracy, 'Validation accuracy:', validation_accuracy)

            saver.save(self.session, os.path.join(LOGDIR, "model.ckpt"), epoch)
            writer.add_summary(s, epoch)

        return

    def result_plotting(self):
        # TODO: complete function definition
        return


def main():
    # reset graph, if done inside tf.Session(),
    # will break because of nested graph
    tf.reset_default_graph()

    with tf.Session() as session:
        cifar = CIFAR10(session, 'cifar')
        # cifar.data_investigation(3, 5, show=True)
        cifar.check_sample_data()
        cifar.build_and_train()


if __name__ == '__main__':
    main()

