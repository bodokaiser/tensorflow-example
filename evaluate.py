import tensorflow as tf

from glob import glob
from model import simple

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30,
    """Number of epochs to train on.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of parallel threads to use.""")

tf.app.flags.DEFINE_integer('patch_size', 30,
    """Patch size on us, mr volume slices.""")
tf.app.flags.DEFINE_integer('batch_size', 30,
    """Batch size of us, mr pairs for training.""")

tf.app.flags.DEFINE_integer('threshold', 1,
    """Remove patches with less or equal reduced sum.""")

tf.app.flags.DEFINE_string('test_glob', 'data/tfrecord/11.tfrecord',
    """Path or wildcard to tfrecord to use for testing.""")
tf.app.flags.DEFINE_string('train_glob', 'data/tfrecord/13.tfrecord',
    """Path or wildcard to tfrecord to use for training.""")

def main(_):
    model = simple.Model(threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs, num_threads=FLAGS.num_threads,
        patch_size=FLAGS.patch_size, batch_size=FLAGS.batch_size)

    us = model.interference()
    loss = model.loss(us)
    train = model.training(loss)
    batch = model.inputs(glob(FLAGS.train_glob))

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0

            while not coord.should_stop():
                _, norm = sess.run([train, loss], feed_dict={
                    model.mr: batch[0].eval(),
                    model.us: batch[1].eval()
                })

                if step % 100 == 0:
                    print('step: {}, norm: {:.0f}'.format(step, norm))

                step += 1

        except tf.errors.OutOfRangeError as error:
            coord.request_stop(error)
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()