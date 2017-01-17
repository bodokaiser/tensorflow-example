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

    def __init__(self, filter_size, filter_num):
        super().__init__()

        self._filter_num = filter_num
        self._filter_size = filter_size

    @property
    def filter_num(self):
        return self._filter_num

    @property
    def filter_size(self):
        return self._filter_size

    def loss(self, re):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(re)

        return loss

    def train(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def conv1(self):
        with tf.name_scope('conv1'):
            weight = _weight([self.filter_size, self.filter_size, 1,
                self.filter_num])
            bias = _bias([self.filter_num])
        return weight, bias

    def conv2(self):
        with tf.name_scope('conv2'):
            weight = _weight([1, 1, self.filter_num, 1])
            bias = _bias([1])

        return weight, bias

    def interference(self, conv1, conv2, batch):
        cnn = _conv(batch, conv1[0]) + conv1[1]
        cnn = _conv(cnn, conv2[0]) + conv2[1]

        return cnn