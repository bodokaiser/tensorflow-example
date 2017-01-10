import tensorflow as tf

import os
import glob
import difflib

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

def convert(mr_source, us_source, target):
    mr_filenames = glob.glob(mr_source)
    us_filenames = glob.glob(us_source)

    def match_ids(filenames, glob):
        diff = lambda a, b: ''.join(d[2:] for d in difflib.ndiff(a, b) if d.startswith('+ '))
        return [diff(glob, f) for f in filenames]

    assert len(us_filenames) == len(mr_filenames)
    ids = match_ids(us_filenames, us_source)
    assert ids == match_ids(mr_filenames, mr_source)

    os.makedirs(os.path.dirname(target), exist_ok=True)

    for i in range(len(us_filenames)):
        mr = minc.read_minc(mr_filenames[i])
        us = minc.read_minc(us_filenames[i])

        assert len(us) == len(mr)
        examples = [tfrecord.encode_example(mr[j], us[j]) for j in range(len(us))]

        tf_filename = target.replace('*', ids[i])
        tfrecord.write_tfrecord(tf_filename, examples)

        print('converted {} and {} to {} ({} slices)'.format(
            mr_filenames[i], us_filenames[i], tf_filename, len(us)))

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