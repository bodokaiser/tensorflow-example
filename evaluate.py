import os
import tensorflow as tf

from model import simple
from hooks import signal

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_filters', 3,
    """Number of filters to use.""")
tf.app.flags.DEFINE_integer('filter_size', 3,
    """Filter size of conv weights.""")

tf.app.flags.DEFINE_integer('logging_steps', 5,
    """Number of steps to do logging.""")
tf.app.flags.DEFINE_integer('summary_steps', 10,
    """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 500,
    """Number of steps to save variable checkpoint.""")

tf.app.flags.DEFINE_string('name', 'train',
    """Name to use as final path or prefix (most test, train, validation).""")
tf.app.flags.DEFINE_string('outdir', '/tmp/mrtous',
    """Path to write summary and variables to.""")
tf.app.flags.DEFINE_string('record', 'data/train.tfrecord',
    """Path to tfrecord to use for training or testing.""")

def init_fn(scaffold, session):
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.outdir)

    if latest_ckpt:
        scaffold.saver.restore(session, latest_ckpt)

def get_global_step():
    graph = tf.get_default_graph()
    return tf.contrib.framework.get_or_create_global_step(graph)

def train(model, records, logdir, vardir):
    mr, us = model.inputs(records)
    us_ = model.interference(mr)
    loss = model.loss(us, us_)
    train = model.train(loss)

    saver = tf.train.Saver()
    scaffold = tf.train.Scaffold(init_fn=init_fn, saver=saver)

    hooks = [
        signal.SignalHandlerHook(),
        tf.train.LoggingTensorHook({
            'norm': loss,
            'step': get_global_step(),
        }, FLAGS.logging_steps),
        tf.train.SummarySaverHook(
            output_dir=logdir,
            save_steps=FLAGS.summary_steps,
            summary_op=tf.summary.merge_all(),
        ),
        tf.train.CheckpointSaverHook(
            checkpoint_dir=vardir,
            save_steps=FLAGS.checkpoint_steps,
            scaffold=scaffold,
        ),
    ]

    with tf.train.MonitoredTrainingSession(
        hooks=hooks, scaffold=scaffold) as session:
        while not session.should_stop():
            session.run(train)

def main(_):
    m = simple.Model(
        num_filters=FLAGS.num_filters,
        filter_size=FLAGS.filter_size)

    train(m, [FLAGS.record],
        os.path.join(FLAGS.outdir, 'train'),
        os.path.join(FLAGS.outdir, ''))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()