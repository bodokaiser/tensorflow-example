import tensorflow as tf

from ioutil import tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('offset', 0,
    """Slice offset to start at.""")
tf.app.flags.DEFINE_integer('length', 100,
    """Number of slices to extract.""")

tf.app.flags.DEFINE_string('input_path', 'data/tfrecord/13.tfrecord',
    """Path to read tfrecord file from.""")
tf.app.flags.DEFINE_string('output_path', 'data/tfrecord/test.tfrecord',
    """Path to write tfrecord file to.""")

def main(_):
    count = 0
    slices = []
    offset = FLAGS.offset-1
    length = FLAGS.length

    for e in tfrecord.iter_tfrecord(FLAGS.input_path):
        if count > offset and count - offset < length:
            slices.append(e)
        count += 1

    tfrecord.write_tfrecord(FLAGS.output_path, slices)

    print('extracted {} slices from {} starting at {} and wrote to {}'.format(
        len(slices), FLAGS.input_path, offset, FLAGS.output_path))

if __name__ == '__main__':
    tf.app.run()