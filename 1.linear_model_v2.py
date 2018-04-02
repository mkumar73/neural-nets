import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)


LOGDIR = "graphs/slm/"

print('Size of dataset:')
print('Training size:{}'.format(len(data.train.labels)))
print('Test size:{}'.format(len(data.test.labels)))
print('Validation size:{}'.format(len(data.validation.labels)))
# print(type(data.train.labels))

# check one hot encoding
# print(data.test.labels[:5,:])

# store label as column vector
data.test.cls = np.array([label.argmax() for label in data.test.labels])
# print(data.test.cls[:5])

# validation labels
data.validation.cls = np.array([label.argmax() for label in data.validation.labels])

# hyperparameters
image_size = 28
image_shape = image_size * image_size
num_classes = 10
learning_rate = 0.005

# function for plotting image

# print(data.train.images[:1,:].shape)

def plot_image(images, true_class):

    # img_len = np.array(len(images))
    # plot_shape = np.reshape(img_len, [-1, 4])
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


# plot_image(data.test.images[:12,:], data.test.cls[:12])


# Tensorflow computational graph
# reset tf graph
tf.reset_default_graph()

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, image_shape], name='input')
    Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
    # take the output as single value rather than one-hot encoding
    y = tf.argmax(Y, axis=1)

with tf.name_scope('fc'):
    weights = tf.Variable(tf.random_normal([image_shape, num_classes], stddev=0.02),name='weights')
    biases = tf.Variable(tf.zeros([num_classes]), name='biases')
    logits = tf.matmul(X, weights) + biases
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("logits", logits)

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

    print('confusion matrix for MNIST data:{}'.format(cm))


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
def plot_weights(sess):

    w = sess.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        image = w[:,i].reshape([image_size, image_size])
        ax.set_xlabel('weight:{0}'.format(i))
        ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
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
def training(is_confusion_matrix=False, is_plot_weights=False, is_plot_cost_accuracy=False):

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
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(session.graph)
        
        # save the model for future use
        saver = tf.train.Saver()

        for epoch in range(1000):
            x_batch, y_batch = data.train.next_batch(batch_size)
            feed_dict_train = {X: x_batch, Y: y_batch}
            cal_cost, _, s = session.run([cost, optimizer, summ], feed_dict=feed_dict_train)

            avg_cost.append(cal_cost)

            # check accuracy on validation set, by chance i have tested on test data so
            # we will use validation set to check the accuracy of the model.
            if epoch % 100 == 0:
                feed_dict_test = {X: data.test.images,
                                  Y: data.test.labels}
                cal_accuracy = session.run(accuracy, feed_dict=feed_dict_test)
                avg_accuracy.append(cal_accuracy)
                writer.add_summary(s, epoch)

                print('Cost after {0} steps:{1}'.format(epoch, cal_cost))
                print('Accuracy after {0} steps:{1}'.format(epoch, cal_accuracy))
            
            if epoch % 500 == 0:
                saver.save(session, os.path.join(LOGDIR, "model.ckpt"), epoch)


        # print confusion matrix on validation data
        if is_confusion_matrix:
            plot_confusion_matrix(session, data.test.cls, feed_dict_test)
        
        # print weights plot
        if is_plot_weights:
            plot_weights(session)

        # plot cost and accuracy graph
        if is_plot_cost_accuracy:
            plot_cost_accuracy(avg_cost, avg_accuracy)

    return



def main():

    # check training status
    training()
    # training(is_plot_cost_accuracy=True)
    # training(is_plot_weights=True, is_plot_cost_accuracy=True)
    # training(is_confusion_matrix=True, is_plot_weights=True, is_plot_cost_accuracy=True)

    # check test accuracy on validation data
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(LOGDIR))
        
        test_accuracy = 0
        test_accuracy = []
        test_cost = []
        feed_dict_val = {X: data.validation.images,
                          Y: data.validation.labels}
        count = 0
        for step in range(2):
            cal_accuracy, cal_cost = session.run([accuracy, cost], feed_dict=feed_dict_val)
            test_accuracy.append(cal_accuracy)
            test_cost.append(cal_cost)
            count +=1

        # print confusion matrix for validation data
        # plot_confusion_matrix(session, data.validation.cls, feed_dict_val)

    # print cost and accuracy calulated for validation data  
    print("Test Accuracy:{0} ".format(np.sum(test_accuracy) / count)) 
    print("Test Cost:{0} ".format(np.sum(test_cost) / count)) 

if __name__ == '__main__':
    main()

