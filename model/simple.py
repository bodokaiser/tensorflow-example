import tensorflow as tf

from model import common

def conv_layer(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape), name='weight')

def bias_variable(shape):
    return tf.Variable(tf.constant(.1, shape=shape), name='bias')

class Simple(common.Base):

    def loss(self, us, us_):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(us-us_)

        return loss

    def train(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def interference(self, mr):
        # TODO: use tensorflow slim for model interference
        with tf.name_scope('conv1'):
            conv1_w = weight_variable([3, 3, 1, 3])
            conv1_b = bias_variable([3])
            conv1 = conv_layer(mr, conv1_w) + conv1_b

        with tf.name_scope('conv2'):
            conv2_w = weight_variable([1, 1, 3, 1])
            conv2 = conv_layer(conv1, conv2_w)

        return conv2