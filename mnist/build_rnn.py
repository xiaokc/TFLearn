# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# training parameters
learning_rate = 0.001
training_steps = 20000
batch_size = 128
display_step = 200

# network parameters
num_input = 28  # MNIST data input(image : 28 * 28)
timesteps = 28  # time steps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes(0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# define weights
weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}


def RNN(x, weight, biases):
    # prepare data shape to match 'rnn' function requirements
    # current data input shape:(batch_size, timesteps, n_input)
    # required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # get lstm cell output
    outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weight['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits=logits)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss=loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()

# start training:
with tf.Session() as sess:
    # run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print "Optimization finish!!"

    # calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing accuracy=", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
