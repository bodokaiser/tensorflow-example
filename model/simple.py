import tensorflow as tf

from model import inputs

def _conv(x, weight):
    return tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME')

def _bias(shape, name='bias'):
    init = tf.constant(.1, tf.float32, shape)
    return _summary(tf.Variable(init, name=name), name)

def _weight(shape, name='weight'):
    init = tf.truncated_normal(shape)
    return _summary(tf.Variable(init, name=name), name)

def _summary(var, name):
    tf.summary.histogram(name, var)
    return var

class Model(inputs.Model):

    def loss(self, us, us_):
        tf.summary.image('us_', us_, max_outputs=1)
        tf.summary.image('us', us, max_outputs=1)

        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(us-us_)
            tf.summary.scalar('loss', loss)

        return loss

    def train(self, loss, step):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss, global_step=step)

        return train

    def interference(self, mr):
        filter_size = self.filter_size
        filter_num = self.num_filters

        with tf.name_scope('conv1'):
            conv1_w = _weight([filter_size, filter_size, 1, filter_num])
            conv1_b = _bias([filter_num])
            conv1 = _conv(mr, conv1_w) + conv1_b

        with tf.name_scope('conv2'):
            conv2_w = _weight([1, 1, filter_num, 1])
            conv2_b = _bias([1])
            conv2 = _conv(conv1, conv2_w) + conv2_b

        return conv2