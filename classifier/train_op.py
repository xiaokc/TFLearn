# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

# 1. 声明批量大小
batch_size = 20

# 2.声明模型数据、占位符和变量
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1, 1]))

# 3. 在计算图中增加矩阵乘法操作
my_output = tf.matmul(x_data, A)

# 4. 损失函数
loss = tf.reduce_mean(tf.square(my_output - y_target))

# 5. 声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss=loss)

sess.run(tf.global_variables_initializer())
# 6. 在训练中通过循环迭代优化模型算法
# 批量训练
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 5 == 0:
        print('Batch Step #' + str(i + 1) + ' A=' + str(sess.run(A)))
        tmp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

        print('Batch Loss=' + str(tmp_loss))
        loss_batch.append(tmp_loss)

# 随机训练
loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [[x_vals[rand_index]]]
    rand_y = [[y_vals[rand_index]]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 5 == 0:
        print('Stochatic Step #' + str(i + 1) + ' A=' + str(sess.run(A)))
        tmp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})

        print('Stochatic Loss=' + str(tmp_loss))
        loss_stochastic.append(tmp_loss)

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochatic')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch')
plt.legend(loc='upper right', prop = {'size':11})
plt.show()
