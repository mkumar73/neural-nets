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
    # plt.show()


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
    	return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape = [None, 32,32,1])
    Y = tf.placeholder(tf.int64, [None])

print('X shape:\t',X.shape)
print('Y shape:\t',Y.shape)

conv1 = conv_relu(X, [3, 3, 1, 8], [8], name='conv1')   
tf.summary.histogram('conv1', conv1)

conv2 = conv_relu(conv1, [3, 3, 8, 16], [16], name='conv2')
tf.summary.histogram('conv2', conv2)

maxpool1 = max_pooling(conv2, name='pool1')

conv3 = conv_relu(maxpool1, [3, 3, 16, 32], [32], name='conv3')
tf.summary.histogram('conv3', conv3)

conv4 = conv_relu(conv3, [3, 3, 32, 64], [64], name='conv4')
tf.summary.histogram('conv4', conv4)

maxpool2 = max_pooling(conv4, name='maxpool2')

# reshape maxpool2 to fit the fully connected layer
fc_ip = tf.reshape(maxpool2, [-1, 8*8*64])
print(fc_ip.shape)

fc = fully_connected(fc_ip, [8*8*64,128], name='fc')
tf.summary.histogram('fc', fc)

logits = output(fc, [128, 10], name='output')

for i in tf.trainable_variables():
    print(i)

#define hyperparameter
LEARNING_RATE = 0.001
epochs = 3
mini_batch_size = 300
plot_step_size = 50

# define loss and accuracy
with tf.name_scope('loss'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    cost = tf.reduce_mean(entropy, name='cost')
    tf.summary.scalar('cost', cost)
    
with tf.name_scope('train'):
    # using Adam optimizer with learning rate of LEARNING_RATE to minimize cost
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

with tf.name_scope('accuracy'):
    prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), Y)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    

# declare training parameters
training_steps = svhn.get_sizes()[0] // mini_batch_size
training_entropies = []
validation_entropies = []
training_accuracies = []
validation_accuracies = []

# training
def training():

	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

	    # merge all summaries
		summ = tf.summary.merge_all()
        # write the summaries
		writer = tf.summary.FileWriter(LOGDIR, session.graph)
        # save the model for future use
		saver = tf.train.Saver()
	    
		step = 0
		last_step = False
		for epoch in range(epochs):
			for images, labels in svhn.get_training_batch(mini_batch_size):
				dict_ = {X: images, Y: labels}
				_, s = session.run([optimizer, summ],feed_dict=dict_)
				t_cost, t_acc = session.run([cost, accuracy], feed_dict=dict_)

				training_entropies.append(t_cost)
				training_accuracies.append(t_acc)

				if step == (training_steps * epochs)-1:
					last_step = True
	            
				if step % plot_step_size == 0 or last_step:
					images, labels = next(svhn.get_validation_batch(0))
					dict_val = {X: images, Y: labels}
					v_cost, v_acc = session.run([cost, accuracy], feed_dict=dict_val)
					
					validation_entropies.append(v_cost)
					validation_accuracies.append(v_acc)
					writer.add_summary(s, step)

				if step % 100 == 0:
					print('Iterations:{2}, Train Acc:{0}, Train cost:{1} '.format(t_acc, t_cost, step))
	            
				step += 1
			saver.save(session, os.path.join(LOGDIR, "model.ckpt"), epoch)
			print('Epoch:{2}, Train Acc:{0}, Train cost:{1} '.format(np.mean(training_accuracies), np.mean(training_entropies), epoch))
			print('Epoch:{2}, Validation Acc:{0}, Validation cost:{1} '.format(np.mean(validation_accuracies), np.mean(validation_entropies), epoch))
	
	return training_entropies, training_accuracies, validation_entropies, validation_accuracies


# plot training results in the graph
def plot_result(t_acc, t_cost, v_acc, v_cost):
	fig_entropy, ax_entropy = plt.subplots()
	fig_entropy.suptitle("Cross Entropy")

	fig_accuracy, ax_accuracy = plt.subplots()
	fig_accuracy.suptitle("Accuracy")

	ax_entropy.cla()
	ax_entropy.plot(training_entropies, label = "Training data")
	ax_entropy.plot(validation_entropies, label = "Validation data")
	ax_entropy.set_xlabel("Training Step")
	ax_entropy.set_ylabel("Entropy")
	ax_entropy.legend()
	fig_entropy.canvas.draw()

	ax_accuracy.cla()
	ax_accuracy.plot(training_accuracies, label = "Training data")
	ax_accuracy.plot(validation_accuracies, label = "Validation data")
	ax_accuracy.set_xlabel("Training Step")
	ax_accuracy.set_ylabel("Accuracy in %")
	ax_accuracy.legend()
	fig_accuracy.canvas.draw()

	plt.show()
	return


# main program
def main():
	t_cost, t_acc, v_cost, v_acc = training()
	plot_result(t_acc, t_cost, v_acc, v_cost)

	with tf.Session() as session:
	    saver = tf.train.Saver()
	    saver.restore(session, tf.train.latest_checkpoint(LOGDIR))
	    
	    test_accuracy = 0
	    for step, (images, labels) in enumerate(svhn.get_test_batch(300)):
	        test_accuracy += session.run(accuracy, feed_dict = {X: images, Y: labels})
    
	print("Test Accuracy: " + str(test_accuracy / step))


# start
if __name__ == '__main__':
	main()

