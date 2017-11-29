import tensorflow as tf

v = tf.constant([[1., 2., 3.], [4., 5., 6.]])

v1 = tf.constant([[5, 2]])
v2 = tf.constant([[2, 4]])
with tf.Session() as sess:
    reduce_mean = tf.reduce_mean(v).eval()
    tf.global_variables_initializer().run()
    print(reduce_mean)
    print(tf.where(tf.greater(v1, v2), x=v1, y=v2).eval())
