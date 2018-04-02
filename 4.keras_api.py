import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from keras.optimizers import Adam


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)


LOGDIR = "graphs/keras/model.keras"

print('Size of dataset:')
print('Training size:\t{}'.format(len(data.train.labels)))
print('Test size:\t{}'.format(len(data.test.labels)))
print('Validation size:\t{}'.format(len(data.validation.labels)))
# print(type(data.train.labels))

print('Train image shape:\t', data.train.images[1].shape)
print('Validation image shape:\t', data.validation.images[1].shape)

# check one hot encoding
# print(data.test.labels[:5,:])

# store label as column vector
data.test.cls = np.array([label.argmax() for label in data.test.labels])
# print(data.test.cls[:5])

# validation labels
data.validation.cls = np.array([label.argmax() for label in data.validation.labels])


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

    dict_ = {X : [image]}

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


# hyperparameters
image_size = 28
image_shape = image_size * image_size
num_classes = 10
learning_rate = 0.005


# Start Keras sequential model building
model = Sequential()

# add input layer
model.add(InputLayer(input_shape=(image_shape,)))

model.add(Reshape([28, 28, 1]))

# 1st conv layer, relu activation and max pooling
model.add(Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu', name='conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# 2nd conv layer, relu activation and max pooling
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# flatten the result
model.add(Flatten())

# Fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))

# output layer
model.add(Dense(num_classes, activation='softmax'))

# adam optimizer
optimizer = Adam(lr=learning_rate)

# model compilation
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# training
model.fit(x=data.train.images, y=data.train.labels, epochs=2, batch_size=128)

# save the model
model.save(LOGDIR)

# evaluation on test set
result = model.evaluate(x=data.test.images, y=data.test.labels)
print('Keras result on test set:\t',result)


# start the main program
def main():

    # check training status
    model = load_model(LOGDIR)
    images = data.test.images[0:9]
    y_pred = model.predict(x=images)
    cls_pred = np.argmax(y_pred, axis=1)
    print('Predicted class for sample test set:\t', cls_pred)
    print('Actual class for sample test set:\t', np.argmax(data.test.labels[0:9], axis=1))
    model.summary()


if __name__ == '__main__':
    main()

