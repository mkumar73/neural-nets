import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)


LOGDIR = "graphs/cnn/v3/"

print('Size of dataset:')
print('Training size:{}'.format(len(data.train.labels)))
print('Test size:{}'.format(len(data.test.labels)))
print('Validation size:{}'.format(len(data.validation.labels)))

print('Train image shape:', data.train.images[1].shape)
print('Validation image shape:', data.validation.images[1].shape)

# store label as column vector
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# validation labels
data.validation.cls = np.array([label.argmax() for label in data.validation.labels])

# hyperparameters
image_size = 28
image_shape = image_size * image_size
num_classes = 10
learning_rate = 0.005


# function for plotting image
def plot_image(images, true_class):

    fig, axes = plt.subplots(3, 4)
    for i, ax in enumerate(np.ravel(axes)):
        image = np.reshape(images[i], [image_size, image_size])
        ax.imshow(image, cmap='binary')
        xlabel = 'True class:{}'.format(true_class[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    return

# reset tf graph
tf.reset_default_graph()

# define helper functions for con2d, fully connected layer and max pooling
def conv_relu(inputs, kernel_shape, bias_shape, name='conv_layer', is_weights=False):
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev=0.01)
        weights = tf.get_variable("weights", kernel_shape, initializer=init)
        biases = tf.get_variable("biases", bias_shape, initializer=init)
        
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')

        if is_weights:
            return tf.nn.relu(conv + biases), weights
        else:
            return tf.nn.relu(conv + biases)


def fully_connected(x, kernel_shape, name='fc', is_relu=False):
    with tf.variable_scope(name):
        init = tf.random_normal_initializer(stddev = 0.01)
        weights = tf.get_variable("weights", kernel_shape, initializer = init)
        biases = tf.get_variable("biases", [kernel_shape[-1]], initializer = init)
        
        fc = tf.matmul(x, weights)

        if is_relu:
            return tf.nn.relu(fc + biases)
        else:
            return tf.nn.tanh(fc + biases)


def output(x, kernel_shape, name='output'):
    with tf.variable_scope(name):
        init = tf.random_normal_initializer(stddev = 0.01)
        weights = tf.get_variable("weights", kernel_shape, initializer = init)
        biases = tf.get_variable("biases", [kernel_shape[-1]], initializer = init)
        
        return tf.matmul(x, weights) + biases


def max_pooling(conv, name='maxpooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# Create DFG

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
    # take the output as single value rather than one-hot encoding
    y = tf.argmax(Y, axis=1)

    print('Input shape:',X.shape)
    print('Label shape:',Y.shape)


conv1, conv1_w = conv_relu(X, [3, 3, 1, 16], [16], name='conv1', is_weights=True)
print('Conv layer 1 shape:', conv1.shape)
tf.summary.histogram('conv1', conv1)

maxpool1 = max_pooling(conv1, name='maxpool1')
print('Max pool 1 shape:', maxpool1)
tf.summary.histogram('maxpool1', maxpool1)

conv2, conv2_w = conv_relu(maxpool1, [3, 3, 16, 32], [32], name='conv2', is_weights=True)
print('Conv layer 1 shape:', conv2.shape)
tf.summary.histogram('conv2', conv2)

maxpool2 = max_pooling(conv2, name='maxpool2')
print('Max pool 1 shape:', maxpool2)
tf.summary.histogram('maxpool2', maxpool2)


fc_input = tf.reshape(maxpool2, [-1, 7*7*32])
print('FC input shape:', fc_input)

fc = fully_connected(fc_input, [7*7*32, 64], name='fc', is_relu=True)
print('FC shape:', fc)
tf.summary.histogram('fc', fc)


logits = output(fc, [64, 10], name='output')
print('Output shape:', logits)
tf.summary.histogram('logits', logits)


# print all trainable variables
for i in tf.trainable_variables():
    print(i)


with tf.name_scope('prediction'):
    # normalize the probabolity value so sum upto 1 for each row.
    y_pred = tf.nn.softmax(logits)
    # get the predicted class for each sample using argmax
    y_pred_cls = tf.argmax(y_pred, axis=1)

with tf.name_scope('cost'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    cost = tf.reduce_mean(entropy)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    # prformance measures
    correct_prediction = tf.equal(y_pred_cls, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# Confusion matrix definition
def plot_confusion_matrix(sess, true_class, dict_):
    predicted_class = sess.run(y_pred_cls, feed_dict=dict_)

    cm = confusion_matrix(y_true=true_class, y_pred=predicted_class)

    print('confusion matrix for MNIST data:\n{}'.format(cm))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return


# plot weights to visualize the optimization and structure of weights learned with time.
def plot_weights(session, weights):
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    filters = w.shape[3]
    grids = math.ceil(math.sqrt(filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(grids, grids)
    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        if i<filters:
            img = w[:, :, 0, i]
            # print('Weights value:',img)
            # print('Weights shape:', img.shape)
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()    
    return


# plot conv layer result to visualize the optimization and structure of conv layers.
def plot_conv_layer(session, layer, image):
    img = np.reshape(image, (28, 28, 1))
    dict_ = {X : [img]}

    conv_result = session.run(layer, feed_dict=dict_) 
    filters = conv_result.shape[3]
    grids = math.ceil(math.sqrt(filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(grids, grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        if i<filters:
            img = conv_result[0, :, :, i]
            # print('Weights value:',img)
            # print('Weights shape:', img.shape)
            ax.imshow(img, interpolation='nearest', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()    
    return


# define cost and accuracy plotting function
def plot_cost_accuracy(cost, accuracy):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(cost)
    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Cost')
    ax2.plot(accuracy)
    ax2.set_xlabel('Training steps')
    ax2.set_ylabel('Accuracy')
    plt.show()
    return


# start training process
def training(is_confusion_matrix=False, is_plot_cost_accuracy=False):

    avg_cost = []
    avg_accuracy = []
    epochs = 3
    batch_size = 100
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        # merge all summaries
        summ = tf.summary.merge_all()

        # write the summaries
        writer = tf.summary.FileWriter(LOGDIR, session.graph)
        
        # save the model for future use
        saver = tf.train.Saver()

        for epoch in range(1000):
            x_batch, y_batch = data.train.next_batch(batch_size)
            x_batch = x_batch.reshape([-1, 28, 28, 1])
            feed_dict_train = {X: x_batch, Y: y_batch}
            cal_cost, _, s = session.run([cost, optimizer, summ], feed_dict=feed_dict_train)
            
            avg_cost.append(cal_cost)

            # check accuracy on validation set, by chance i have tested on test data so
            # we will use validation set to check the accuracy of the model.
            if epoch % 100 == 0:
                val_images = data.validation.images.reshape([-1, 28, 28, 1])
                feed_dict_val = {X: val_images,
                                  Y: data.validation.labels}
                cal_accuracy = session.run(accuracy, feed_dict=feed_dict_val)
                avg_accuracy.append(cal_accuracy)

                print('Cost after {0} steps:{1}'.format(epoch, cal_cost))
                print('Accuracy after {0} steps:{1}'.format(epoch, cal_accuracy))
            
            if epoch % 500 == 0:
                saver.save(session, os.path.join(LOGDIR, "model.ckpt"), epoch)
                writer.add_summary(s, epoch)

        # print confusion matrix on validation data
        if is_confusion_matrix:
            plot_confusion_matrix(session, data.validation.cls, feed_dict_val)
        

        # plot cost and accuracy graph
        if is_plot_cost_accuracy:
            plot_cost_accuracy(avg_cost, avg_accuracy)

    return


# start the main program
def main():

    # check training status
    # training()
    # training(is_plot_cost_accuracy=True)
    # training(is_confusion_matrix=True)
    training(is_plot_cost_accuracy=True, is_confusion_matrix=True)

    # print weights for conv layers and show conv layer  
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(LOGDIR))
        
        print('Weights for first conv layer:')
        plot_weights(session, conv1_w)

        print('Weights for secind conv layer:')
        plot_weights(session, conv2_w)

        img = data.test.images[7]

        print('First conv layer:')
        plot_conv_layer(session, conv1, img)

        print('Second conv layer:')
        plot_conv_layer(session, conv2, img)

    # check test accuracy on test data
        test_accuracy = 0
        test_accuracy = []
        test_cost = []
        test_images = data.test.images.reshape([-1, 28, 28, 1])
        feed_dict_test = {X: test_images,
                          Y: data.test.labels}
        count = 0
        for step in range(2):
            cal_accuracy, cal_cost = session.run([accuracy, cost], feed_dict=feed_dict_test)
            test_accuracy.append(cal_accuracy)
            test_cost.append(cal_cost)
            count +=1

        # print confusion matrix for test data
        plot_confusion_matrix(session, data.test.cls, feed_dict_test)

    # print cost and accuracy calulated for validation data  
    print("Test Accuracy:{0} ".format(np.sum(test_accuracy) / count)) 
    print("Test Cost:{0} ".format(np.sum(test_cost) / count)) 

if __name__ == '__main__':
    main()

