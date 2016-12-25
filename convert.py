import tensorflow as tf

from os import path, makedirs
from glob import glob
from ioutil import minc
from ioutil import tfrecord
from difflib import ndiff

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('us_glob', 'data/minc/*_us.mnc',
    """Path or wildcard to read us volumes as minc2.""")
tf.app.flags.DEFINE_string('mr_glob', 'data/minc/*_mr.mnc',
    """Path or wildcard to read mr volumes as minc2.""")
tf.app.flags.DEFINE_string('tf_glob', 'data/tfrecord/*.tfrecord',
    """Path or wildcard to write volumes as tfrecord.""")

def match_ids(filenames, glob):
    diff = lambda a, b: ''.join(d[2:] for d in ndiff(a, b) if d.startswith('+ '))
    return [diff(glob, f) for f in filenames]

def main(_):
    us_filenames, mr_filenames = glob(FLAGS.us_glob), glob(FLAGS.mr_glob)
    assert len(us_filenames) == len(mr_filenames)
    ids = match_ids(us_filenames, FLAGS.us_glob)
    assert ids == match_ids(mr_filenames, FLAGS.mr_glob)

    makedirs(path.dirname(FLAGS.tf_glob), exist_ok=True)

    for i in range(len(us_filenames)):
        us = minc.read_minc(us_filenames[i])
        mr = minc.read_minc(mr_filenames[i])

        tf_filename = FLAGS.tf_glob.replace('*', ids[i])
        tfrecord.write_tfrecord(tf_filename, us, mr)

        print('converted {} and {} to {} ({} slices)'.format(
            us_filenames[i], mr_filenames[i], tf_filename, len(us)))

if __name__ == '__main__':
    tf.app.run()