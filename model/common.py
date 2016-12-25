import tensorflow as tf

from ioutil import tfrecord

def extract_patches(image, patch_size):
    images = tf.expand_dims(image, 0)
    patches = tf.extract_image_patches(images, [1, patch_size, patch_size, 1],
        [1, patch_size, patch_size, 1], [1, 1, 1, 1], 'VALID')
    return tf.reshape(patches, [-1, patch_size, patch_size, 1])

def filter_patches(patches, threshold):
    return tf.where(tf.greater(tf.reduce_sum(patches, [1, 2]), threshold))[:, 0]

class Base(object):

    def __init__(self, threshold, batch_size, patch_size,
        num_epochs, num_threads):
        self._threshold = threshold
        self._batch_size = batch_size
        self._patch_size = patch_size
        self._num_epochs = num_epochs
        self._num_threads = num_threads

    @property
    def threshold(self):
        return self._threshold

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_threads(self):
        return self._num_threads

class Model(Base):

    def loss(self, us, us_):
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(us-us_)

        return loss

    def train(self, loss):
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer().minimize(loss)

        return train

    def placeholder(self):
        shape = [None, self.patch_size, self.patch_size, 1]

        with tf.name_scope('placeholder'):
            us = tf.placeholder(tf.float32, shape)
            mr = tf.placeholder(tf.float32, shape)

        return us, mr

    def inputs(self, filenames):
        queue = tf.train.string_input_producer(filenames,
            num_epochs=self.num_epochs)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader(options=tfrecord.TFRECORD_OPTIONS)
            _, example = reader.read(queue)

        with tf.name_scope('decode'):
            mr_slice, us_slice = tfrecord.decode_example(example)

        with tf.name_scope('patch'):
            mr_patches = extract_patches(mr_slice, self.patch_size)
            us_patches = extract_patches(us_slice, self.patch_size)

        with tf.name_scope('filter'):
            # TODO: run filter also on mr_patches and merge result
            indices = filter_patches(us_patches, self.threshold)
            mr_patches = tf.gather(mr_patches, indices)
            us_patches = tf.gather(us_patches, indices)

        return tf.train.batch([mr_patches, us_patches], self.batch_size,
            num_threads=self.num_threads, enqueue_many=True)