import tensorflow as tf

import os

from ioutil import minc
from ioutil import tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('offset', 0,
    """Slice offset to start at.""")
tf.app.flags.DEFINE_integer('length', 0,
    """Number of slices to extract.""")

def count(source):
    with tf.Session() as session:
        count = sum([1 for e in tfrecord.iter_tfrecord(source)])

    print('counted {} slices in {}'.format(count, source))

def convert(mr_filename, us_filename, tf_filename):
    tf_dirname = os.path.dirname(tf_filename)

    if tf_dirname != '':
        os.makedirs(tf_dirname, exist_ok=True)

    mr = minc.read_minc(mr_filename)
    us = minc.read_minc(us_filename)

    tfrecord.write_tfrecord(tf_filename,
        [tfrecord.encode_example(mr[i], us[i]) for i in range(len(mr))])

    print('converted {} and {} to {} ({} slices)'.format(
        mr_filename, us_filename, tf_filename, len(us)))

def partition(source, target):
    count = 0
    slices = []

    for e in tfrecord.iter_tfrecord(source):
        if count > FLAGS.offset:
            if FLAGS.length > 0:
                if count - FLAGS.offset < FLAGS.length + 1:
                    slices.append(e)
                else:
                    break
            else:
                slices.append(e)
        count += 1

    tfrecord.write_tfrecord(target, slices)

    print('extracted {} slices from {} starting at {} and wrote to {}'.format(
        len(slices), source, FLAGS.offset, target))

def main(args):
    if args[1] == 'count':
        return count(args[2])
    if args[1] == 'convert':
        return convert(*args[2:5])
    if args[1] == 'partition':
        return partition(*args[2:4])

    print('Unknown subcommand use either count, convert or partition.')

if __name__ == '__main__':
    tf.app.run()