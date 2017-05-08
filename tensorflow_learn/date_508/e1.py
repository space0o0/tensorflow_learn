# coding=utf-8
# 滑动平均模型

import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)

step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer();
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))  # v1的值改为5，计算衰减率min{0.99,(1+step)/(10+step)}=0.1
    sess.run(maintain_averages_op)  # v1的滑动平均值更新为 0.1*0+(1-0.1)*5=4.5
    print (sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print (sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)
    print (sess.run([v1, ema.average(v1)]))
