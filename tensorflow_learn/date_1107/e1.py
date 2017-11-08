import tensorflow as tf

matrix1 = tf.constant([[3, 1]])
matrix2 = tf.constant([[2, 2, 2],
                       [1, 1, 1]])
product = tf.matmul(matrix1, matrix2)
# print(matrix1)
# print(matrix2)
# print(sess.run(matrix1))
# print(sess.run(matrix2))
# print(sess.run(product))


# 模拟单个神经网络的计算
# create weight
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# create input x shape(1,2)
x = tf.constant([[0.7, 0.9]])

# 先计算隐藏层的数据，x的输出，y的输入
a = tf.matmul(x, w1)
# 计算y
y = tf.matmul(a, w2)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(y.eval())
