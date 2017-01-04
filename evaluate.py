import tensorflow as tf

import os
import glob
import hooks
import model
import utils
import runner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train',
    """Either train or test.""")

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

tf.app.flags.DEFINE_string('name', 'train',
    """Name to use as final path or prefix (most test, train, validation).""")
tf.app.flags.DEFINE_string('outdir', '/tmp/mrtous',
    """Path to write summary and variables to.""")
tf.app.flags.DEFINE_string('records', 'data/test.tfrecord',
    """Path or wildcard to tfrecord to use for testing.""")

def train(model, records, logdir, vardir):
    mr, us = model.inputs(records)
    us_ = model.interference(mr)
    loss = model.loss(us, us_)
    train = model.train(loss)

    r = runner.Runner(vardir)
    r.add_hook(hooks.SignalHandlerHook())
    r.add_hook(hooks.LoggingHook({'norm': loss}, 100))
    r.add_hook(hooks.StopAfterNStepsHook(FLAGS.max_steps))
    r.add_hook(hooks.SummarySaverHook(logdir, FLAGS.summary_steps))
    r.add_hook(hooks.CheckpointSaverHook(vardir, FLAGS.checkpoint_steps,
        r.scaffold))

    with r.monitored_session() as session:
        while not session.should_stop():
            session.run(train)

def test(model, records, logdir, vardir):
    mr, us = model.inputs(records)
    us_ = model.interference(mr)
    loss = model.loss(us, us_)

    r = runner.Runner(vardir)
    r.add_hook(hooks.StepCounterHook())
    r.add_hook(hooks.SignalHandlerHook())
    r.add_hook(hooks.LoggingHook({'norm': loss}, 100))
    r.add_hook(hooks.StopAfterNStepsHook(FLAGS.max_steps))
    r.add_hook(hooks.SummarySaverHook(logdir, FLAGS.summary_steps))

    with r.monitored_session() as session:
        while not session.should_stop():
            session.run(loss)

def main(_):
    m = model.SimpleModel(
        threshold=FLAGS.threshold,
        num_epochs=FLAGS.num_epochs,
        num_threads=FLAGS.num_threads,
        filter_size=FLAGS.filter_size,
        patch_size=FLAGS.patch_size,
        batch_size=FLAGS.batch_size)

    if FLAGS.mode == 'train':
        run = train
    elif FLAGS.mode == 'test':
        run = test
    else:
        raise ValueError('Unknown mode.')

    run(m, glob.glob(FLAGS.records),
        os.path.join(FLAGS.outdir, FLAGS.name),
        os.path.join(FLAGS.outdir, ''))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()