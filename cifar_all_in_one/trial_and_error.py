import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# print(x_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(y_train.shape)

print(y_test[:10])
#
sample_image = x_train[:10]
sample_label = y_train[:10]
# print(sample_label[5])
# print(sample_label.shape)

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

print(label_to_word[int(sample_label[5])])
