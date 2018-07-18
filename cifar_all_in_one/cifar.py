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

    def __init__(self, data='cifar', lr=0.01, batch_size=64, epochs=5, init=None):

        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._init = init

    def _load_data(self, data):
        """
        :param data: dataset name
        :return: dataset and labels for training and test
        """
        data.lower()
        if data=='cifar':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        else:
            logging.error('Incorrect dataset name given')
        return x_train, y_train, x_test, y_test

    def _train_val_split(self):
        """
        :return: training, validation and test set data
        """
        x_train, y_train, x_test, y_test = self._load_data()

        x_train, x_validation = x_train[5000:], x_train[:5000]
        y_train, y_validation = y_train[5000:], y_train[:5000]
        return x_train, x_validation, x_test, y_train, y_validation, y_test

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

    def data_investigation(self, x, y, show=False):
        """
        function to investigate data using plots
        :param x: subplot parameter
        :param y: subplot parameter
        :return: plot figures with labels from training data
        """
        n_images = x*y

        x_train, y_train, x_test, y_test = self._load_data(self.data)

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
            ax.imshow(sample_image[i,:,:,:])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            # ax.set_axis_off()
            ax.set_title(str(int(sample_label[i])) + ':' + label_to_word[int(sample_label[i])])
            if show:
                plt.show()
        return

    def fully_connected(self, input, kernel_shape, act_fn='relu', name='fc', output=False):
        """

        :param input:
        :param kernal_shape:
        :param act_fn:
        :param name:
        :param init:
        :param output:
        :return:
        """
        with tf.name_scope(name):
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


    def cond2d_relu(self, input, kernal_shape, bias_shape, name='conv', is_weights=False):
        """

        :param input:
        :param kernal_shape:
        :param bias_shape:
        :param name:
        :param is_weights:
        :return:
        """
        with tf.name_scope(name):
            if self._init == 'xavier':
                init = tf.contrib.layers.xavier_initializer
            else:
                init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name='weight', shape=kernal_shape, initializer=init)
            biases = tf.get_variable(name='bias', shape=bias_shape, initializer=init)

            conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

            if not is_weights:
                return tf.nn.relu(conv + biases)
            else:
                return tf.nn.relu(conv + biases), weights


    def max_pooling(self):
        # TODO: complete function definition
        return

    def batch_norm(self):
        # TODO: complete function definition
        return

    def build_and_train(self):
        # TODO: complete function definition
        return

    def result_plotting(self):
        # TODO: complete function definition
        return

cifar = CIFAR10('cifar')
cifar.data_investigation(3, 5, show=True)



