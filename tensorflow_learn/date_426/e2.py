# coding=utf-8
from __future__ import print_function
import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数 weight
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数和反向传播
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

sess = tf.InteractiveSession()

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    print("===========================初始权重=======================")
    print(sess.run(w1))
    print(sess.run(w2))
    # print (X)
    print("===========================初始权重=======================")
    # print(Y)

STEPS = 5000
for i in range(STEPS):
    # 每次选取batch_size个样本
    start = (i * batch_size) % batch_size
    end = min(start + batch_size, dataset_size)
    # 选取样本训练神经网络并更新数据
    sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    if i % 1000 == 0:
        # 每隔一段时间计算交叉熵并输出
        total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
        print("alter %d training steps,cross entropy on all data is %g" % (i, total_cross_entropy))

print("===========================训练的权重=======================")
print(sess.run(w1))
print(sess.run(w2))
print("===========================训练的权重=======================")
