import tensorflow as tf
from tensorflow.python import debug as tf_debug

a = tf.constant(10.0, name='a')
b = tf.Variable(5.0, name='b')
c = tf.Variable(3.0, name='c')

x = tf.multiply(a, b)
y = tf.add(c, x)

session = tf.Session()
session = tf_debug.TensorBoardDebugWrapperSession(session, 'localhost:7007')

init = tf.global_variables_initializer()

session.run(init)
session.run(y)	