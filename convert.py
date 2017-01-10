import tensorflow as tf

import os

from glob import glob
from ioutil import minc
from ioutil import tfrecord
from difflib import ndiff

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('us_path', 'data/minc/*_us.mnc',
    """Path to read us volumes as MINC-2.0""")
tf.app.flags.DEFINE_string('mr_path', 'data/minc/*_mr.mnc',
    """Path to read mr volumes as MINC-2.0""")
tf.app.flags.DEFINE_string('tf_path', 'data/tfrecord/*.tfrecord',
    """Path to write volumes as tfrecord.""")

def match_ids(filenames, glob):
    diff = lambda a, b: ''.join(d[2:] for d in ndiff(a, b) if d.startswith('+ '))
    return [diff(glob, f) for f in filenames]

def main(_):
    mr_filenames = glob(FLAGS.mr_path)
    us_filenames = glob(FLAGS.us_path)
    assert len(us_filenames) == len(mr_filenames)
    ids = match_ids(us_filenames, FLAGS.us_path)
    assert ids == match_ids(mr_filenames, FLAGS.mr_path)

    os.makedirs(os.path.dirname(FLAGS.tf_path), exist_ok=True)

    for i in range(len(us_filenames)):
        mr = minc.read_minc(mr_filenames[i])
        us = minc.read_minc(us_filenames[i])

        assert len(us) == len(mr)
        examples = [tfrecord.encode_example(mr[j], us[j]) for j in range(len(us))]

        tf_filename = FLAGS.tf_path.replace('*', ids[i])
        tfrecord.write_tfrecord(tf_filename, examples)

        print('converted {} and {} to {} ({} slices)'.format(
            mr_filenames[i], us_filenames[i], tf_filename, len(us)))

if __name__ == '__main__':
    tf.app.run()