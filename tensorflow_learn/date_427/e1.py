from __future__ import print_function
import tensorflow as tf

v = tf.constant(10, tf.float32)
a = tf.constant([[1., 2., 3.], [7., 8., 9.]])
tf.InteractiveSession()
print(tf.reduce_mean(a).eval())
