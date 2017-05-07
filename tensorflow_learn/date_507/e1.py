# coding=utf-8
# 过拟合问题 tf实战 p89

import tensorflow as tf


def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda1)(var))  # 加入正则
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))


