import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from model import simple
from image import transform
from ioutil import tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('filter_size', 3,
    """Filter size of conv weights.""")
tf.app.flags.DEFINE_integer('patch_size', 30,
    """Patch size on us, mr volume slices.""")

tf.app.flags.DEFINE_string('vardir', '/tmp/mrtous',
    """Path to variables to load.""")
tf.app.flags.DEFINE_string('records', 'data/train.tfrecord',
    """Path to tfrecord to use as input.""")

def main(_):
    height = transform.compatible_dim(tfrecord.SLICE_HEIGHT, FLAGS.patch_size)
    width = transform.compatible_dim(tfrecord.SLICE_WIDTH, FLAGS.patch_size)

    model = simple.Model(
        num_epochs=1,
        num_threads=1,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=int((height/FLAGS.patch_size)*(width/FLAGS.patch_size)))
    mr, us = model.inputs(glob(FLAGS.records), filter_value=None)

    #saver = tf.train.Saver()
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer()
        ])

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            mr_image = sess.run(transform.patches_to_image(mr, height, width))

            plt.imshow(mr_image[:, :, 0], cmap='gray')
            plt.show()
        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()