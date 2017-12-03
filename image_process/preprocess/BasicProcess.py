# coding=utf-8

import tensorflow as tf

img_path = "../images/cat.jpeg"

reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer([img_path]))
image0 = tf.image.decode_jpeg(value)
image = tf.expand_dims(image0, 0)
image_summary = tf.summary.image('original image', image)
histogram_summary = tf.summary.histogram('image hist', image)
e = tf.reduce_mean(image)
scalar_summary = tf.summary.scalar('image mean', e)

resize_image = tf.image.resize_images(image, [256,256], method=tf.image.ResizeMethod.AREA)
img_resize_summary = tf.summary.image('image resize', resize_image)

merged = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    print sess.run(init_op)
    threads = tf.train.start_queue_runners(sess)
    img = image.eval()
    print img.shape

    summary_writer = tf.summary.FileWriter('../tmp/tensorboard', sess.graph)
    summary_all = sess.run(merged)
    summary_writer.add_summary(summary_all, 0)
    summary_writer.close()