import tensorflow as tf

from image import transform
from ioutil import tfrecord

class Model(object):

    def __init__(self, threshold, batch_size, patch_size, filter_size,
        num_epochs, num_threads):
        self._threshold = threshold
        self._batch_size = batch_size
        self._patch_size = patch_size
        self._filter_size = filter_size
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
    def filter_size(self):
        return self._filter_size

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def num_threads(self):
        return self._num_threads

    def inputs(self, filenames):
        queue = tf.train.string_input_producer(filenames,
            num_epochs=self.num_epochs)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader(options=tfrecord.TFRECORD_OPTIONS)
            _, example = reader.read(queue)

        with tf.name_scope('decode'):
            mr_slice, us_slice = tfrecord.decode_example(example)

        with tf.name_scope('resize'):
            h = transform.compatible_dim(tfrecord.SLICE_HEIGHT, self.patch_size)
            w = transform.compatible_dim(tfrecord.SLICE_WIDTH, self.patch_size)
            mr_slice = tf.image.resize_image_with_crop_or_pad(mr_slice, h, w)
            us_slice = tf.image.resize_image_with_crop_or_pad(us_slice, h, w)

        with tf.name_scope('patch'):
            mr_patches = transform.image_to_patches(mr_slice, self.patch_size,
                self.patch_size)
            us_patches = transform.image_to_patches(us_slice, self.patch_size,
                self.patch_size)

        if self.threshold is not None:
            with tf.name_scope('filter'):
                indices = transform.filter_indices(us_patches, self.threshold)
                mr_patches = tf.gather(mr_patches, indices)
                us_patches = tf.gather(us_patches, indices)

        return tf.train.batch([mr_patches, us_patches], self.batch_size,
            self.num_threads, self.num_threads*self.batch_size*3,
            enqueue_many=True)