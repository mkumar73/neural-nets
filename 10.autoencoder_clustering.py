import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from tensorflow.contrib.factorization.python.ops import clustering_ops
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform 2D 28x28 matrix to 3D (28x28x1) matrix
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# inputs have to be between [0, 1]
x_train /= 255
x_test /= 255

model = Sequential()

# encoder layers
# 1st convolution layer
model.add(Conv2D(16, (3, 3) , padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

# 2nd convolution layer
model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

# Decoder layers
# 3rd convolution layer
model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

# 4th convolution layer
model.add(Conv2D(16,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

# output layer
model.add(Conv2D(1,(3, 3), padding='same'))
model.add(Activation('sigmoid'))

model.summary()

# model.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# model.fit(x_train, x_train,
#           epochs=3,
#           validation_data=(x_test, x_test))
#
# model.save('ae_clustering.hd5')

model = load_model('ae_clustering.hd5')

restored_imgs = model.predict(x_test)

for i in range(3):
    plt.imshow(x_test[i].reshape(28, 28))
    label = 'Original Image:' + str(y_test[i])
    plt.title(label)
    plt.gray()
    plt.show()

    plt.imshow(restored_imgs[i].reshape(28, 28))
    label = 'Restorted Image:' + str(y_test[i])
    plt.title(label)
    plt.gray()
    plt.show()


for i in range(len(model.layers)):
    print('Layer:', i, ". ", model.layers[i].output.get_shape())

# layer[7] is activation_3 (Activation), it is compressed representation
get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[7].output])
compressed = get_3rd_layer_output([x_test])[0]

# layer[7] is size of (None, 7, 7, 2). this means 2 different 7x7 sized matrixes. We will flatten these matrixes.
compressed = compressed.reshape(10000, 7*7*2)


unsupervised_model = tf.contrib.learn.KMeansClustering(
                                                10, #num of clusters
                                                distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE,
                                                initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT)

def return_train_input():
    data = tf.constant(compressed, tf.float32)
    return (data, None)


unsupervised_model.fit(input_fn=return_train_input, steps=1000)

clusters = unsupervised_model.predict(input_fn=return_train_input)

index = 0
for i in clusters:
    current_cluster = i['cluster_idx']
    features = x_test[index]

    if index < 200 and current_cluster == 9:
        plt.imshow(x_test[index].reshape(28, 28))
        label = 'Original Image:' + str(y_test[index])
        plt.title(label)
        label = 'cluster id:' + str(current_cluster)
        plt.xlabel(label)
        plt.gray()
        plt.show()
    index = index + 1

clusters = unsupervised_model.predict(input_fn=return_train_input)

index = 0
cluster_dict = {}
for i in clusters:
    #     current_cluster = i['cluster_idx']
    #     print(current_cluster.)
    #     print(i['cluster_idx'])
    if i['cluster_idx'] == 9:
        index += 1
        cluster_dict[i['cluster_idx']] = index

print(cluster_dict)


cluster_distribution = [1286, 1182, 725, 778, 922, 1111, 896, 1053, 991, 1056]

x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, cluster_distribution, color='gray')
plt.xlabel("Clusters")
plt.ylabel("#datapoints in each cluster")
plt.title("Data points in each cluster")

plt.xticks(x_pos, x)

plt.show()


