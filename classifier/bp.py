import numpy as np
import tensorflow as tf

sess = tf.Session()

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.0, 100)
x_data = tf.placeholder(shape=[1],dtype=tf.float32)
