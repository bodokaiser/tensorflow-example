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
    """Adds simple one layer CNN to InputsModel."""

    def loss(self, us, _us):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(us-_us)

            tf.summary.scalar('loss', loss)

        return loss

    def train(self, loss):
        with tf.name_scope('train'):
            step = tf.Variable(0, name='step', trainable=False)
            train = tf.train.AdamOptimizer().minimize(loss, global_step=step)

        return train, step

    def interference(self, mr):
        fs = self._filter_size

        with tf.name_scope('conv1'):
            conv1_w = _weight([fs, fs, 1, fs])
            conv1_b = _bias([fs])
            conv1 = _conv(mr, conv1_w) + conv1_b

        with tf.name_scope('conv2'):
            conv2_w = _weight([1, 1, fs, 1])
            conv2_b = _bias([1])
            conv2 = _conv(conv1, conv2_w) + conv2_b

        return conv2