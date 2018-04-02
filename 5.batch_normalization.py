# Batch normalization implementation with MNIST data.

# BN is applied before activation function so that 
# the input to the activation function across each 
# training batch has a mean of 0 and a variance of 1.

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
LOGDIR = "graphs/bn/"


# Placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Hyper parameters
# Small epsilon value for the BN transform
epsilon = 1e-3
lr = 0.01

# Layer 1 without BN
with tf.name_scope('layer_1'):
    w1 = tf.Variable(tf.random_normal([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    
    z1 = tf.matmul(x,w1)+b1
    l1 = tf.nn.sigmoid(z1)
    
    tf.summary.histogram('w1', w1)
    tf.summary.histogram('z1', z1)
    tf.summary.histogram('l1', l1)


# Layer 1 with BN
with tf.name_scope('layer_1_BN'):    
    w1_BN = tf.Variable(tf.random_normal([784, 100]))

    # Note that pre-batch normalization bias is ommitted. The effect of this bias would be
    # eliminated when subtracting the batch mean. Role of the bias is performed
    # by the new beta variable
    z1_BN = tf.matmul(x, w1_BN)

    # Calculate batch mean and variance
    batch_mean1, batch_var1 = tf.nn.moments(z1_BN, [0])

    # Apply the initial batch normalizing transform
    z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

    # Create two new parameters, scale and beta (shift)
    scale1 = tf.Variable(tf.ones([100]))
    beta1 = tf.Variable(tf.zeros([100]))

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    BN1 = scale1 * z1_hat + beta1
    l1_BN = tf.nn.sigmoid(BN1)

    tf.summary.histogram('w1_BN', w1_BN)
    tf.summary.histogram('BN1', BN1)
    tf.summary.histogram('l1_BN', l1_BN)


# Layer 2 without BN
with tf.name_scope('layer_2'):
    w2 = tf.Variable(tf.random_normal([100, 100]))
    b2 = tf.Variable(tf.zeros([100]))
    
    z2 = tf.matmul(l1, w2) + b2
    l2 = tf.nn.sigmoid(z2)
    
    tf.summary.histogram('w2', w2)
    tf.summary.histogram('z2', z2)
    tf.summary.histogram('l2', l2)


# Layer 2 with BN, using Tensorflows built-in BN function
with tf.name_scope('layer_2_BN'):    
    w2_BN = tf.Variable(tf.random_normal([100, 100]))
    z2_BN = tf.matmul(l1_BN, w2_BN)

    batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])

    scale2 = tf.Variable(tf.ones([100]))
    beta2 = tf.Variable(tf.zeros([100]))

    BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)

    l2_BN = tf.nn.sigmoid(BN2)

    tf.summary.histogram('w2_BN', w1_BN)
    tf.summary.histogram('BN2', BN2)
    tf.summary.histogram('l2_BN', l2_BN)

# Softmax
with tf.name_scope('logits'):
    w3 = tf.Variable(tf.random_normal([100, 10]))
    b3 = tf.Variable(tf.zeros([10]))

    logits = tf.matmul(l2, w3) + b3
    y  = tf.nn.softmax(logits)

    tf.summary.histogram('w3', w3)
    tf.summary.histogram('logits', logits)


with tf.name_scope('logits_BN'):    
    w3_BN = tf.Variable(tf.random_normal([100, 10]))
    b3_BN = tf.Variable(tf.zeros([10]))

    logits_BN = tf.matmul(l2_BN, w3_BN) + b3_BN
    y_BN  = tf.nn.softmax(logits_BN)

    tf.summary.histogram('w3_BN', w3_BN)
    tf.summary.histogram('logits_BN', logits_BN)

# Loss, optimizer and predictions
with tf.name_scope('cost'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('without BN', cross_entropy)

with tf.name_scope('cost_BN'):
    cross_entropy_BN = -tf.reduce_sum(y_ * tf.log(y_BN))
    tf.summary.scalar('with BN', cross_entropy_BN)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

with tf.name_scope('train_BN'):
    train_step_BN = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy_BN)


with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('without BN', accuracy)

with tf.name_scope('accuracy_BN'):    
    correct_prediction_BN = tf.equal(tf.argmax(y_BN, 1),tf.argmax(y_, 1))
    accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN, tf.float32))
    tf.summary.scalar('with BN', accuracy_BN)

# training the network
zs, BNs, acc, acc_BN = [], [], [], []

init = tf.global_variables_initializer()

with tf.Session() as session:
    
    session.run(init)

    # merge all summaries
    summ = tf.summary.merge_all()

    # write the summaries
    writer = tf.summary.FileWriter(LOGDIR, session.graph)

    # save the model for future use
    saver = tf.train.Saver()
    
    print('Training in progress.......')
    
    for i in range(10001):
        batch = mnist.train.next_batch(100)
        
        _, _, s = session.run([train_step, train_step_BN, summ], feed_dict={x: batch[0], y_: batch[1]})
        
        dict_ = {x: mnist.test.images, y_: mnist.test.labels}

        if i % 1000 is 0:
            res = session.run([accuracy,accuracy_BN,z2,BN2],feed_dict=dict_)
            acc.append(res[0])
            acc_BN.append(res[1])
            zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
            BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN2 over the entire test set
            print('Steps: {2}, Accuracy with BN: {1} and without BN: {1}'.format(res[0], res[1], i))

            writer.add_summary(s, i)
        
        if i % 5000 == 0:
                saver.save(session, os.path.join(LOGDIR, "model.ckpt"), i)

zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)


# plot the result
fig, ax = plt.subplots()

ax.plot(range(0,len(acc)*50,50),acc, label='Without BN')
ax.plot(range(0,len(acc)*50,50),acc_BN, label='With BN')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.8,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.show()


# visualize the effect of neurons in 2nd layer
fig, axes = plt.subplots(5, 2, figsize=(6,12))
fig.tight_layout()

for i, ax in enumerate(axes):
    ax[0].set_title("Without BN")
    ax[1].set_title("With BN")
    ax[0].plot(zs[:,i])
    ax[1].plot(BNs[:,i])

    plt.show()








