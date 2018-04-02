import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

print('Size of dataset:')
print('Training size:{}'.format(len(data.train.labels)))
print('Test size:{}'.format(len(data.test.labels)))
print('Validation size:{}'.format(len(data.validation.labels)))

# print(type(data.train.labels))
# check one hot encoding
# print(data.test.labels[:5,:])

# store label as column vector
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[:5])

# hyperparameters
image_size = 28
image_shape = image_size * image_size
num_classes = 10
learning_rate = 0.0005

# function for plotting image
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

X = tf.placeholder(tf.float32, [None, image_shape], name='input')
Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
y_true_cls = tf.placeholder(tf.int64, [None], name='trueclass')

weights = tf.Variable(tf.random_normal([image_shape, num_classes], stddev=0.02),name='weights')
biases = tf.Variable(tf.zeros([num_classes]), name='biases')

logits = tf.matmul(X, weights) + biases

# normalize the probabolity value so sum upto 1 for each row.
y_pred = tf.nn.softmax(logits)

# get the predicted class for each sample using argmax
y_pred_cls = tf.argmax(y_pred, axis=1)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start tf session
avg_cost = []
avg_accuracy = []
epochs = 3
batch_size = 100

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for epoch in range(1000):
        x_batch, y_batch = data.train.next_batch(batch_size)
        feed_dict_train = {X: x_batch, Y: y_batch}
        cal_cost, _ = session.run([cost, optimizer], feed_dict=feed_dict_train)

        avg_cost.append(cal_cost)

        feed_dict_test = {X: data.test.images,
                          Y: data.test.labels,
                          y_true_cls: data.test.cls}
        if epoch % 50 == 0:
            cal_accuracy = session.run(accuracy, feed_dict=feed_dict_test)
            avg_accuracy.append(cal_accuracy)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(avg_cost)
ax1.set_xlabel('Training steps')
ax1.set_ylabel('Cost')
ax2.plot(avg_accuracy)
ax2.set_xlabel('Training steps')
ax2.set_ylabel('Accuracy')

plt.show()






