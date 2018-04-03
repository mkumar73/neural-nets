# In this session we are implementing VGG achitecture for SVHN dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import struct
import os

DATADIR = "../data/svhn/"
LOGDIR = "../logs/vgg/"


# Helper class for data preprocessing
class SVHN():
    def __init__(self, directory = "/data"):
        self._directory = directory
        
        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._test_data = np.array([])
        self._test_labels = np.array([])
        
        self._load_traing_data()
        self._load_test_data()
        
        np.random.seed(0)
        samples_n = self._training_labels.shape[0]
        random_indices = np.random.choice(samples_n, samples_n // 10, replace = False)
        np.random.seed()
        
        self._validation_data = self._training_data[random_indices]
        self._validation_labels = self._training_labels[random_indices]
        self._training_data = np.delete(self._training_data, random_indices, axis = 0)
        self._training_labels = np.delete(self._training_labels, random_indices)
    
    
    def _load_traing_data(self):
        self._training_data, self._training_labels = self._load_data("train_32x32.mat")        
    
    def _load_test_data(self):
        self._test_data, self._test_labels = self._load_data("test_32x32.mat")
    
    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def _load_data(self, file):
        path = os.path.join(self._directory, file)
        
        mat = scio.loadmat(path)
        data = np.moveaxis(mat["X"], 3, 0)
        data = self._rgb2gray(data)
        data = data.reshape(data.shape + (1,))
        
        labels = mat["y"].reshape(mat["y"].shape[0])
        labels[labels == 10] = 0
        
        return data, labels
    
    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)
    
    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)
    
    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)
    
    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]
        
        if batch_size <= 0:
            batch_size = samples_n
        
        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]
    
    
    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n


# check data from data directory
svhn = SVHN(DATADIR)
print('Size of training, validation and test set:\t',svhn.get_sizes())

# data investigation
image, label = next(svhn.get_training_batch(25))
print('Size of training batch images:',image.shape)
print('Labels of training batch images:',label)

# plot the images to investigate
fig, axs = plt.subplots(3, 4)
for i, ax in enumerate(np.reshape(axs, [-1])):
    ax.imshow(image[i,:,:,0], cmap='gray')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(label[i])


# Construction phase
# reset all variables if necessary
tf.reset_default_graph()

# utility functions
# define con_relu and max pooling to simplify the process
variance_epsilon = 1e-3
init = tf.random_normal_initializer(stddev = 0.01)
init_conv = tf.truncated_normal_initializer(stddev=0.01)


def batch_norm(inputs, is_training):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])

        return tf.nn.batch_normalization(inputs,
            batch_mean, batch_var, beta, scale, variance_epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, variance_epsilon)


def conv_relu(inputs, kernel_shape, bias_shape, name='conv'):
    # Create variable named "weights".
	with tf.variable_scope(name):
	    weights = tf.get_variable("weights", kernel_shape, initializer=init_conv)

	    # Create variable named "biases".
	    biases = tf.get_variable("biases", bias_shape, initializer=init_conv)
	    
	    conv = tf.nn.conv2d(inputs, weights,
	        strides=[1, 1, 1, 1], padding='SAME')
	    conv_bn = batch_norm(conv, is_training=True)
	    return tf.nn.relu(conv_bn + biases)


def fully_connected(x, kernel_shape, name='fc'):
	with tf.variable_scope(name):
	    weights = tf.get_variable("weights", kernel_shape, initializer=init)
	    biases = tf.get_variable("biases", [kernel_shape[-1]], initializer=init)
	    fc = tf.matmul(x, weights)
	    fc = batch_norm(fc, is_training=True)
	    return tf.nn.tanh(fc + biases)


def output(x, kernel_shape, name='output'):
	with tf.variable_scope(name):
	    weights = tf.get_variable("weights", kernel_shape, initializer=init)
	    biases = tf.get_variable("biases", [kernel_shape[-1]], initializer=init)
	    return tf.matmul(x, weights) + biases


def max_pooling(conv, name='pooling'):
    with tf.variable_scope(name):
    return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')


with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape = [None, 32,32,1])
    Y = tf.placeholder(tf.int64, [None])

print('X shape:\t',X.shape)
print('Y shape:\t',Y.shape)




