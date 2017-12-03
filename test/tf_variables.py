#coding=utf-8
import tensorflow as tf
import scipy.io


def read_vgg():
    vgg_rawnet = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_rawnet['layers'][0]


def counter():
    # 创建一个变量
    state = tf.Variable(0, name='counter')

    # 创建一个op，其作用是使state增加1
    op = tf.constant(1)
    new_value = tf.add(state, op)
    update = tf.assign(state, new_value)

    # 启动图后，变量必须先经过初始化，(init)op初始化
    # 首先必须增加一个"初始化"op到图中
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)

        #打印'state'的初始值
        print sess.run(state)

        # 运行 op, 更新'state',并打印'state'
        for _ in range(3):
            sess.run(update)
            print sess.run(state)

if __name__ == '__main__':
    counter()
    # read_vgg()