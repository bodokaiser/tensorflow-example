import tensorflow as tf

import glob
import hooks
import model
import runner

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

tf.app.flags.DEFINE_integer('max_steps', 0,
    """Number of steps to run training.""")
tf.app.flags.DEFINE_integer('summary_steps', 1000,
    """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 1000,
    """Number of steps to save variable checkpoint.""")

tf.app.flags.DEFINE_string('logdir', '/tmp/mrtous',
    """Path to write summary events to.""")
tf.app.flags.DEFINE_string('records', 'data/test.tfrecord',
    """Path or wildcard to tfrecord to use for testing.""")

def main(_):
    m = model.SimpleModel(
        threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs,
        num_threads=FLAGS.num_threads,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=FLAGS.batch_size)

    mr, us = m.inputs(glob.glob(FLAGS.records))
    us_ = m.interference(mr)
    loss = m.loss(us, us_)
    train = m.train(loss)

    r = runner.Runner(FLAGS.logdir,
        save_summary_steps=FLAGS.summary_steps,
        save_checkpoint_steps=FLAGS.checkpoint_steps)
    r.add_hook(hooks.SignalHandlerHook())
    r.add_hook(hooks.LoggingHook({'norm': loss}, 100))

    if FLAGS.max_steps > 0:
        r.add_hook(tf.train.StopAtStepHook(FLAGS.max_steps))

    with r.monitored_session() as session:
        while not session.should_stop():
            session.run(train)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()