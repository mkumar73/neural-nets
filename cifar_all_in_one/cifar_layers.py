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

    def __init__(self, session: tf.Session(), data='cifar', lr=0.001,
                 batch_size=64, epochs=5, early=False, optimizer='adam',
                 bn=False, init_std='xavier'):
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
        self._init = init_std

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

        # to keep the pixel value between 0 and 1
        self.x_train = self.x_train.astype(np.float32)/255.0
        self.x_test = self.x_test.astype(np.float32)/255.0

        self.y_train = self.y_train.astype(np.int64)
        self.y_test = self.y_test.astype(np.int64)
        return

    def _train_val_split(self, _index=5000):
        """
        # split the data and also reshape the label as vectors instead of 1D array
        :param _index: index for slicing
        :return: training, validation and test set data
        """
        self._data_preprocessing()

        x_train, x_validation = self.x_train[_index:], self.x_train[:_index]
        y_train, y_validation = self.y_train[_index:].reshape(-1,), self.y_train[:_index].reshape(-1,)
        x_test, y_test = self.x_test, self.y_test.reshape(-1,)
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
        print(y_test[:1])
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
            x_batch, y_batch = x[batch_idx, :, :, :], y[batch_idx]
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


    def batch_norm(self):
        # TODO: complete function definition
        return

    def build_and_train(self):
        """
        :return:
        """

        # Network building
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input')
            y = tf.placeholder(tf.int64, shape=[None], name='label')

        if self._init == 'xavier':
            init = tf.contrib.layers.xavier_initializer()
        elif self._init == 'normal':
            init = tf.truncated_normal_initializer(stddev=0.01)

        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv2d(X, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                     padding='same', kernel_initializer=init, name='conv1')
            tf.summary.histogram('conv1', conv1)

            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), name='pool1')
            # tf.summary.histogram('pool1', pool1)

        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                     padding='same', kernel_initializer=init, name='conv2')
            tf.summary.histogram('conv2', conv2)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), name='pool2')
            tf.summary.histogram('pool2', pool2)

        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                     padding='same', kernel_initializer=init, name='conv3')
            tf.summary.histogram('conv3', conv3)

            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), name='pool1')
            # tf.summary.histogram('pool1', pool1)

        with tf.name_scope('conv4'):
            conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=(3, 3), strides=(1, 1),
                                     padding='same', kernel_initializer=init, name='conv4')
            tf.summary.histogram('conv4', conv4)

            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2, 2), strides=(2, 2), name='pool4')
            tf.summary.histogram('pool2', pool2)

        with tf.name_scope('flatten'):
            fc_input = tf.layers.flatten(pool4)

        with tf.name_scope('fc'):
            fc = tf.layers.dense(inputs=fc_input, units=64, activation=tf.nn.relu,
                                 kernel_initializer=init, name='fc1')
            tf.summary.histogram('fc', fc)

        with tf.name_scope('logit'):
            logit = tf.layers.dense(inputs=fc, units=10, activation=None, name='output')
            tf.summary.histogram('output', logit)

        # print all trainable variables
        for i in tf.trainable_variables():
            print(i)

        with tf.name_scope('cost'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y)
            cost = tf.reduce_mean(entropy)
            tf.summary.scalar('cost', cost)

        with tf.name_scope('accuracy'):
            # normalize the probability value so sum to 1 for each row.
            # get the predicted class for each sample using argmax, index presents class
            y_pred = tf.argmax(tf.nn.softmax(logit), axis=1)
            # performance measures
            prediction = tf.equal(y_pred, y)
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('optimizer'):
            self.optimizer.lower()
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(cost)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)
        # Network build ends

        # training process started
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

            batch_accuracy = self.session.run(accuracy, feed_dict={X: x_batch, y: y_batch})
            validation_accuracy = self.session.run(accuracy, feed_dict={X: x_validation, y: y_validation})

            print('Epoch:', epoch+1, 'Batch accuracy:', batch_accuracy, 'Validation accuracy:', validation_accuracy)
            # print('Epoch:', epoch, 'Batch accuracy:', batch_accuracy)

            # write the summary for every epoch
            writer.add_summary(s, epoch)
            # write model ckpts for last epoch
            if epoch == self.epochs-1:
                saver.save(self.session, os.path.join(LOGDIR, "model.ckpt"), epoch)

        return

    def result_plotting(self):
        # TODO: complete function definition
        return


def main():
    # reset graph, if done inside tf.Session(),
    # will break because of nested graph
    tf.reset_default_graph()

    with tf.Session() as session:
        cifar = CIFAR10(session, 'cifar', batch_size=128, epochs=5, optimizer='adam', init_std='xavier')
        # cifar.data_investigation(3, 5, show=True)
        # cifar.check_sample_data()
        cifar.build_and_train()


if __name__ == '__main__':
    main()

