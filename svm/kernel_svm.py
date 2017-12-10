# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# 生成模拟数据。两个同心圆数据，每个不同的环代表的不同的类，确保类只有-1和1
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=0.5, noise=0.1)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

batch_size = 250
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

# 创建高斯核函数
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
# 线性核函数可以表示为：my_kernel = tf.matmul(x_data, tf.transpose(x_data))


# 声明对偶问题
model_output = tf.matmul(b, my_kernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, second_term))

# 创建预测函数和准确度函数
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                      tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
# 如果是线性预测核函数，此处应为：pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
prediction = tf.sign(prediction_output - tf.reduce_sum(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), dtype=tf.float32))

# 创建优化器函数
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
batch_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    batch_accuracy.append(temp_acc)

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1))
        print('Loss = ' + str(temp_loss))

    # x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
    # y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    #
    # grid_points = np.c_[xx.ravel(), yy.ravel()]
    # [grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points})
    # grid_predictions = grid_predictions.reshape(xx.shape)
    #
    #
    # plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha = 0.8)
    # plt.plot(class1_x, class1_y, 'ro', label = 'Class 1')
    # plt.plot(class2_x, class2_y, 'kx', label = 'Class 2')
    # plt.legend(loc = 'lower right')
    # plt.ylim([-1.5, 1.5])
    # plt.xlim([-1.5, 1.5])
    # plt.show()

plt.plot(batch_accuracy, 'k-', label = 'Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

