import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    tf.get_variable("v", initializer=1.1, dtype=tf.float32)

g2 = tf.Graph()
with g2.as_default():
    tf.get_variable("v", initializer=2., dtype=tf.float32)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print (sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print (sess.run(tf.get_variable("v")))


