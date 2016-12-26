import tensorflow as tf
import tensorflow.contrib.slim as slim

from model import inputs

class Model(inputs.Model):
    """Simple one layer CNN model on top of """

    def loss(self, us):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(self.us-us)

        tf.summary.scalar('loss', loss)

        return loss

    def training(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def interference(self):
        net = slim.conv2d(self.mr, 3, [3, 3], scope='conv1')
        net = slim.conv2d(net, 1, [1, 1], scope='conv2')

        return net