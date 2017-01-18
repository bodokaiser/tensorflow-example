import tensorflow as tf

from ioutil import tfrecord

def image_to_patches(image, shape):
    images = tf.expand_dims(image, 0)
    patches = tf.extract_image_patches(images, shape, shape,
        [1, 1, 1, 1], 'VALID', name='extract')
    return tf.reshape(patches, [-1, *shape[1:]])

def filter_sum_greater(tensors, value):
    return tf.where(tf.greater(tf.reduce_sum(tensors, [1, 2]), value))[:, 0]

class Model(object):

    def __init__(self, num_threads=8):
        self._num_threads = num_threads

    @property
    def num_threads(self):
        return self._num_threads

    def inputs(self, filenames):
        queue = tf.train.string_input_producer(filenames)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader(options=tfrecord.TFRECORD_OPTIONS)
            _, example = reader.read(queue)

        with tf.name_scope('decode'):
            mr, us = tfrecord.decode_example(example)

        return mr, us

    def images(self, filenames):
        mr, us = self.inputs(filenames)

        return tf.expand_dims(mr, 0), tf.expand_dims(us, 0)

    def patches(self, filenames, patch_size=7, batch_size=128, threshold=0):
        mr, us = self.inputs(filenames)

        with tf.name_scope('patch'):
            shape = [1, patch_size, patch_size, 1]
            mr_patches = image_to_patches(mr, shape)
            us_patches = image_to_patches(us, shape)

        with tf.name_scope('filter'):
            indices = filter_sum_greater(us_patches, threshold)
            mr_patches = tf.gather(mr_patches, indices)
            us_patches = tf.gather(us_patches, indices)

        return tf.train.shuffle_batch([mr_patches, us_patches], batch_size,
            num_threads=self.num_threads, min_after_dequeue=batch_size,
            capacity=self.num_threads*batch_size*3, enqueue_many=True)