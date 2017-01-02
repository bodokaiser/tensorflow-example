import tensorflow as tf

from model import base
from ioutil import tfrecord

def extract_patches(image, shape):
    patches = tf.extract_image_patches(tf.expand_dims(image, 0), shape, shape,
        [1, 1, 1, 1], 'VALID')
    return tf.reshape(patches, [-1, *shape[1:]])

def filter_patches_indices(patches, value):
    indices = tf.where(tf.greater(tf.reduce_sum(patches, [1, 2]), value))
    return indices[:, 0]

class Model(base.Model):
    """Adds input pipeline to BaseModel."""

    def placeholder(self):
        shape = [None, self.patch_size, self.patch_size, 1]

        with tf.name_scope('placeholder'):
            us = tf.placeholder(tf.float32, shape)
            mr = tf.placeholder(tf.float32, shape)

        return mr, us

    def inputs(self, filenames):
        queue = tf.train.string_input_producer(filenames,
            num_epochs=self.num_epochs)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader(options=tfrecord.TFRECORD_OPTIONS)
            _, example = reader.read(queue)

        with tf.name_scope('decode'):
            mr_slice, us_slice = tfrecord.decode_example(example)

        with tf.name_scope('patch'):
            shape = [1, self.patch_size, self.patch_size, 1]
            mr_patches = extract_patches(mr_slice, shape)
            us_patches = extract_patches(us_slice, shape)

        with tf.name_scope('filter'):
            # TODO: run filter also on mr_patches and merge result
            indices = filter_patches_indices(us_patches, self.threshold)
            mr_patches_filtered = tf.gather(mr_patches, indices)
            us_patches_filtered = tf.gather(us_patches, indices)

        return tf.train.batch([mr_patches_filtered, us_patches_filtered],
            self.batch_size, num_threads=self.num_threads, enqueue_many=True)