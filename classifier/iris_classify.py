# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()

# 将山鸢尾花标记为1，其余标记为0,训练只使用两种特征：花瓣长度和花瓣宽度
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# 声明批量训练大小、数据占位符和模型变量
batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
my_mul = tf.matmul(x2_data, A)
my_add = tf.add(my_mul, b)
my_output = tf.subtract(x1_data, my_add)

# sigmoid交叉熵损失函数
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

# 声明优化器方法
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 1000次迭代训练
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])

    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})

    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ',A=' + str(sess.run(A)) + ',b=' + str(sess.run(b)))
