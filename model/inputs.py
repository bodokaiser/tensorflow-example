import tensorflow as tf

from model import base
from ioutil import tfrecord

class Model(base.Model):

    def inputs(self, filenames):
        queue = tf.train.string_input_producer(filenames,
            num_epochs=self.num_epochs)

        with tf.name_scope('reader'):
            reader = tf.TFRecordReader(options=tfrecord.TFRECORD_OPTIONS)
            _, example = reader.read(queue)

        with tf.name_scope('decode'):
            mr, us = tfrecord.decode_example(example)

        return tf.train.batch([mr, us], self.batch_size, self.num_threads,
            self.num_threads*self.batch_size*3)