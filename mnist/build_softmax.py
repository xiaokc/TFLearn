# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", [None, 784])  # n * 784
y_ = tf.placeholder("float", [None, 10])  # n * 10

W = tf.Variable(tf.zeros([784, 10]))  # 784 * 10
b = tf.Variable(tf.zeros([10]))

# softmax 预测值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# TensorFlow自动使用反向传播算法，按照梯度下降算法以0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化创建的变量
init = tf.initialize_all_variables()

# 模型评估
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(init)  # 在Session中启动模型，并初始化变量
    # 训练模型1000次
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print 'iter {0}, train accuracy is {1}'.format(i, train_accuracy)

    test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print "test accuracy is {0}".format(test_accuracy)
