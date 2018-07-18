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
import  matplotlib.pyplot as plt
import os


LOGDIR = "logs/"


class CIFAR10():

    def __init__(self, data='cifar'):

        self.data = data

    def _load_data(self, data):
        """
        :param data: dataset name
        :return: dataset and labels for training and test
        """

        if data=='cifar':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_train, y_train, x_test, y_test

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
        :return: plot figures with labels
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
            ax.imshow(sample_image[i])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_axis_off()
            ax.set_title(str(int(sample_label[i])) + ':' + label_to_word[int(sample_label[i])])
            if show:
                plt.show()
        return


cifar = CIFAR10('cifar')
# cifar.data_investigation(3, 5)



