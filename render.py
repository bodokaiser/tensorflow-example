import numpy as np
import scipy as sp
import tensorflow as tf

import re
import os
import glob

from scipy import misc
from model import simple
from image import transform
from ioutil import tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('steps', 1,
    """Steps to render and save images.""")

tf.app.flags.DEFINE_string('vardir', '/tmp/mrtous/patch5-filter3-threshold0',
    """Path to variables to load.""")
tf.app.flags.DEFINE_string('outdir', '/tmp/mrtous',
    """Path to write image files to.""")
tf.app.flags.DEFINE_string('records', 'data/train.tfrecord',
    """Path to tfrecord to use as input.""")

def extract_params_from_path(path):
    fragments = path.split('/')[-1].split('-')
    extract_alpha = lambda s: ''.join(re.findall('[a-z]', s))
    extract_number = lambda s: int(re.findall('\d+', s)[0])
    return {extract_alpha(p): extract_number(p) for p in fragments}

def main(_):
    params = extract_params_from_path(FLAGS.vardir)
    records = glob.glob(FLAGS.records)

    height = transform.compatible_dim(tfrecord.SLICE_HEIGHT, params['patch'])
    width = transform.compatible_dim(tfrecord.SLICE_WIDTH, params['patch'])

    model = simple.Model(
        threshold=None,
        num_epochs=1,
        num_threads=1,
        filter_size=params['filter'],
        patch_size=params['patch'],
        batch_size=int((height/params['patch'])*(width/params['patch'])),
    )
    mr, us = model.inputs(records)
    us_ = model.interference(mr)

    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    checkpt = tf.train.latest_checkpoint(FLAGS.vardir)

    with tf.Session() as sess:
        sess.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer()
        ])

        if checkpt:
            saver.restore(sess, checkpt)

        threads = tf.train.start_queue_runners(sess, coord)

        try:
            step = 0

            while not coord.should_stop() and step < FLAGS.steps:
                us_img, re_img = sess.run([
                    transform.patches_to_image(us, height, width),
                    transform.patches_to_image(us_, height, width),
                ])

                for i in range(height):
                    for j in range(width):
                        if us_img[i, j] == 0:
                            re_img[i, j] = 0

                misc.toimage(us_img[:, :, 0], cmin=0.0, cmax=1.0).save(
                    os.path.join(FLAGS.outdir, 'us{}.png'.format(step)))
                misc.toimage(re_img[:, :, 0], cmin=0.0, cmax=1.0).save(
                    os.path.join(FLAGS.outdir, 're{}.png'.format(step)))

                print('rendered slice {}'.format(step))

                step += 1
        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()