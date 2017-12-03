# coding=utf-8
import numpy as np
import tensorflow as tf


def do_mul():
    x_vals = np.array([1., 3., 5., 7., 9.])
    x_data = tf.placeholder(tf.float32)
    m_const = tf.constant(4.)
    my_op = tf.multiply(x_data, m_const)
    sess = tf.Session()
    for x_val in x_vals:
        print(sess.run(my_op, feed_dict={x_data: x_val}))


def do_shape():
    # 创建数据和占位符
    my_array = np.array([[1., 3., 5., 7., 9.],
                         [-2., 0., 2., 4., 6.],
                         [-6., -3., 0., 3., 6.]])  # 3*5
    x_vals = np.array([my_array, my_array + 1])
    x_data = tf.placeholder(tf.float32)

    # 穿件矩阵乘法和加法中要用到的常量矩阵
    m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])  # 5*1
    m2 = tf.constant([[2.]])  # 1*1
    a1 = tf.constant([[10.]])  # 1*1

    # 声明操作，表示成计算图
    prod1 = tf.multiply(x_data, m1)
    prod2 = tf.multiply(prod1, m2)
    add1 = tf.add(prod2, a1)

    # 通过计算图赋值
    with tf.Session() as sess:
        for x_val in x_vals:
            print sess.run(add1, feed_dict={x_data: x_val})


if __name__ == '__main__':
    do_mul()
