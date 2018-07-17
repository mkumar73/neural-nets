# class based approach to for MNIST classification
# using tf.layers.dense instead of contrib


import tensorflow as tf
import numpy as np


class MNIST_DENSE():


    def __init__(self, data='mnist', input_size=28, lr=0.01,
                 batch_size=64, epochs=10):
        """

        :param data:
        :param input_size:
        :param lr:
        :param batch_size:
        :param epochs:
        """

        self.data = data
        self.input_size = input_size
        self.image_size = input_size*input_size
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def _load_data(self):
        """

        :return:
        """
        if self.data=='mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        else:
            print('Only implmented for MNIST as of now.!!')

        return

    def _data_preprocessing(self):
        """

        :return:
        """
        self._load_data()

        x_train = self.x_train.astype(np.float32).reshape(-1, self.image_size)/255.0
        x_test = self.x_test.astype(np.float32).reshape(-1, self.image_size)/255.0

        y_train = self.y_train.astype(np.int32)
        y_test = self.y_test.astype(np.int32)

        return x_train, x_test, y_train, y_test

    def _train_test_split(self, _index = 5000):
        """

        :param _index:
        :return:
        """
        x_train, x_test, y_train, y_test = self._data_preprocessing()

        x_train, x_validation = x_train[5000:], x_train[:5000]
        y_train, y_validation = y_train[5000:], y_train[:5000]

        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def check_sample_data(self):
        """

        :return:
        """
        train, val, test, _, _, _ = self._train_test_split(_index=5000)
        print('Size of train, validation and test set:\n')
        print(train.shape)
        print(val.shape)
        print(test.shape)
        print('Sample data:\n')
        print(train[:2])

        return


mnist = MNIST_DENSE('mnist', 28, 0.01, 64, 10)

# mnist.check_sample_data()

