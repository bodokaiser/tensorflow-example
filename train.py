import tensorflow as tf

from glob import glob
from model import simple

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30,
    """Number of epochs to train on.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of parallel threads to use.""")

tf.app.flags.DEFINE_integer('filter_size', 3,
    """Filter size of conv weights.""")
tf.app.flags.DEFINE_integer('patch_size', 30,
    """Patch size on us, mr volume slices.""")
tf.app.flags.DEFINE_integer('batch_size', 30,
    """Batch size of us, mr pairs for training.""")

tf.app.flags.DEFINE_integer('threshold', 1,
    """Remove patches with less or equal reduced sum.""")

tf.app.flags.DEFINE_string('logdir', '/tmp/mrtous',
    """Path to write summary events to.""")
tf.app.flags.DEFINE_string('records', 'data/test.tfrecord',
    """Path or wildcard to tfrecord to use for testing.""")

def main(_):
    model = simple.Model(
        threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs,
        num_threads=FLAGS.num_threads,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=FLAGS.batch_size)

    mr, us = model.placeholder()
    us_ = model.interference(mr)
    loss = model.loss(us, us_)

    train, global_step = model.train(loss)
    batch = model.inputs(glob(FLAGS.records))

    tf.summary.image('us_', us_, max_outputs=1)
    tf.summary.image('us', us, max_outputs=1)

    summary_op = tf.summary.merge_all()

    with tf.train.MonitoredTrainingSession() as session:
        while not session.should_stop():
            _, _, step, norm, summary = session.run(
                [train, us_, global_step, loss, summary_op],
                feed_dict={
                    mr: batch[0].eval(session=session),
                    us: batch[1].eval(session=session),
                })
            print('step {}, norm {:.0f}'.format(step, norm))

if __name__ == '__main__':
    tf.app.run()