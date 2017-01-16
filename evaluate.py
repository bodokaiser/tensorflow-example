import os
import argparse
import tensorflow as tf

from model import simple
from hooks import signal

DEFAULT_LOGGING_STEPS = 10
DEFAULT_SUMMARY_STEPS = 5

def train(model, records, outdir):
    step = tf.contrib.framework.get_or_create_global_step(
        tf.get_default_graph())

    mr, us = model.inputs(records)
    us_ = model.interference(mr)
    loss = model.loss(us, us_)
    train = model.train(loss, step)

    hooks = [
        tf.train.LoggingTensorHook({
            'norm': loss,
            'step': step,
        }, DEFAULT_LOGGING_STEPS),
        tf.train.SummarySaverHook(
            output_dir=outdir,
            save_steps=DEFAULT_SUMMARY_STEPS,
            summary_op=tf.summary.merge_all(),
        ),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as session:
        while not session.should_stop():
            session.run(train)

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    model = simple.Model(
        num_filters=args.num_filters,
        filter_size=args.filter_size)

    train(model, [args.record], args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluate model on a given dataset')
    parser.add_argument('--record',
        help='tfrecord file to use as for training')
    parser.add_argument('--outdir', required=True,
        help='directory to write summary events to')
    parser.add_argument('--num_filters', type=int,
        help='number of convolutional filters')
    parser.add_argument('--filter_size', type=int,
        help='height and width of convolutional filter')
    parser.set_defaults(record='data/train.tfrecord',
        num_filters=3, filter_size=3)

    main(parser.parse_args())