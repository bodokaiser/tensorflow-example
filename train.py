import tensorflow as tf

import glob

from model import SimpleModel
from hooks import SignalHandlerHook

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30,
    """Number of epochs to train on.""")
tf.app.flags.DEFINE_integer('num_threads', 4,
    """Number of parallel threads to use.""")
tf.app.flags.DEFINE_integer('num_steps', 0,
    """Number of steps to run training.""")

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

def init_fn(scaffold, session):
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.logdir)

    if latest_ckpt:
        scaffold.saver.restore(session, latest_ckpt)

def main(_):
    model = SimpleModel(
        threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs,
        num_threads=FLAGS.num_threads,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=FLAGS.batch_size)

    mr, us = model.inputs(glob.glob(FLAGS.records))
    us_ = model.interference(mr)
    loss = model.loss(us, us_)
    train = model.train(loss)

    tf.summary.image('us_', us_, max_outputs=1)
    tf.summary.image('us', us, max_outputs=1)

    saver = tf.train.Saver()
    scaffold = tf.train.Scaffold(
        saver=saver,
        init_fn=init_fn,
    )

    hooks = [
        tf.train.LoggingTensorHook({
            'step': tf.train.get_global_step(tf.get_default_graph()),
            'norm': loss,
        }, every_n_iter=100),
        tf.train.SummarySaverHook(
            save_steps=1000,
            output_dir=FLAGS.logdir,
            summary_op=tf.summary.merge_all()
        ),
        tf.train.CheckpointSaverHook(
            save_steps=1000,
            scaffold=scaffold,
            checkpoint_dir=FLAGS.logdir,
        ),
        SignalHandlerHook(),
    ]

    if FLAGS.num_steps > 0:
        hooks.append(tf.train.StopAtStepHook(FLAGS.num_steps))

    with tf.train.MonitoredTrainingSession(scaffold=scaffold, hooks=hooks) as session:
        while not session.should_stop():
            session.run(train)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()