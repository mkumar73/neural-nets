import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

print(y_test[:10])