# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)  # -1到1之间的等差数列
target = tf.constant(0)

# L2正则损失函数
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# L1 正则损失函数
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Huber损失函数
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals) / delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals) / delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

# 重新给x_vals和target赋值，保存返回值并绘制
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500, ], 1.)

# hinge损失函数
hinge_y_vals = tf.maximum(0., 1., -tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

# Cross-entropy loss
xentropy_y_vals = tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

# sigmoid cross-entropy loss
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

# weighted cross entropy loss
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets,weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

# softmax cross entropy loss
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_entropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits,target_dist)
print(sess.run(softmax_entropy))

unscaled_logits = tf.constant([[1.,3.,10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits, sparse_target_dist)
print(sess.run(sparse_xentropy))

x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')



