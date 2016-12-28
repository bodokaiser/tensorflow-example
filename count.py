import tensorflow as tf

from ioutil import tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('path', 'data/tfrecord/13.tfrecord',
    """Path to tfrecord to count slices.""")

def main(_):
    path = FLAGS.path

    with tf.Session() as session:
        count = sum([1 for e in tfrecord.iter_tfrecord(path)])

    print('counted {} slices in {}'.format(count, path))

if __name__ == '__main__':
    tf.app.run()