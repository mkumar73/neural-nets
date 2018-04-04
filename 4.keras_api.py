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

LOGDIR = "../logs/keras/model.keras"

print('Size of dataset:')
print('Training size:\t{}'.format(len(data.train.labels)))
print('Test size:\t{}'.format(len(data.test.labels)))
print('Validation size:\t{}'.format(len(data.validation.labels)))
# print(type(data.train.labels))

print('Train image shape:\t', data.train.images[1].shape)
print('Validation image shape:\t', data.validation.images[1].shape)

# store label as column vector
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# validation labels
data.validation.cls = np.array([label.argmax() for label in data.validation.labels])

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

