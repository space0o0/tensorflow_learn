import tensorflow as tf

v=tf.constant([[1.,2.,3.],[4.,5.,6.]])
with tf.Session() as sess:
    reduce_mean=tf.reduce_mean(v).eval()
    tf.global_variables_initializer().run()
    print(reduce_mean)